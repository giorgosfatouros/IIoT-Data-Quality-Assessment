# Copyright 2015 Lukas Lalinsky
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import collections
from pyLeanxcale.types import javaTypetoNative, nativeToParamType, datetimeToTimezoneUtc, timezoneFromDatetimeUtc, \
    LXARRAY
from pyLeanxcale.errors import ProgrammingError, ArrayTypeException
from pyphoenix.common_pb2 import Rep, TypedValue
import datetime

__all__ = ['Cursor', 'ColumnDescription', 'DictCursor']

logger = logging.getLogger(__name__)

# TODO see note in Cursor.rowcount()
MAX_INT = 2 ** 64 - 1

ColumnDescription = collections.namedtuple('ColumnDescription', 'name type_code display_size internal_size precision scale null_ok')
"""Named tuple for representing results from :attr:`Cursor.description`."""


class Cursor(object):
    """Database cursor for executing queries and iterating over results.

    You should not construct this object manually, use :meth:`Connection.cursor() <phoenixdb.connection.Connection.cursor>` instead.
    """

    arraysize = 1
    """
    Read/write attribute specifying the number of rows to fetch
    at a time with :meth:`fetchmany`. It defaults to 1 meaning to
    fetch a single row at a time.
    """

    itersize = 2000
    """
    Read/write attribute specifying the number of rows to fetch
    from the backend at each network roundtrip during iteration
    on the cursor. The default is 2000.
    """

    lastrowid = None

    def __init__(self, connection, id=None):
        self._connection = connection
        self._id = id
        self._prepared_sql = None
        self._signature = None
        self._column_data_types = []
        self._frame = None
        self._pos = None
        self._closed = False
        self.arraysize = self.__class__.arraysize
        self.itersize = self.__class__.itersize
        self._updatecount = -1

    def __del__(self):
        if not self._connection._closed and not self._closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._closed:
            self.close()

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    next = __next__

    def close(self):
        """Closes the cursor.
        No further operations are allowed once the cursor is closed.

        If the cursor is used in a ``with`` statement, this method will
        be automatically called at the end of the ``with`` block.
        """
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        if self._id is not None:
            self._connection._client.close_statement(self._connection._id, self._id)
            self._id = None
            self._prepared_sql = None
        self._signature = None
        self._column_data_types = []
        self._frame = None
        self._pos = None
        self._closed = True

    @property
    def closed(self):
        """Read-only attribute specifying if the cursor is closed or not."""
        return self._closed

    @property
    def description(self):
        if self._signature is None:
            return None
        description = []
        for column in self._signature.columns:
            description.append(ColumnDescription(
                column.column_name,
                column.type.name,
                column.display_size,
                None,
                column.precision,
                column.scale,
                None if column.nullable == 2 else bool(column.nullable),
            ))
        return description

    def _set_id(self, id):
        if self._id is not None and self._id != id:
            self._prepared_sql = None
            self._connection._client.close_statement(self._connection._id, self._id)
        self._id = id

    def _set_type(self, types, attribute_string, return_list):
        for column in types:
            return_list.append(javaTypetoNative(getattr(column, attribute_string)))

    def _set_signature(self, signature):
        self._signature = signature
        self._column_data_types = []
        self._parameter_data_types = []
        if signature is None:
            return

        self._set_type(signature.columns, 'column_class_name', self._column_data_types)

        self._set_type(signature.parameters, 'class_name', self._parameter_data_types)


    def _set_frame(self, frame):
        self._frame = frame
        self._pos = None

        if frame is not None:
            if frame.rows:
                self._pos = 0
            #elif not frame.done:
                #print('got an empty frame, but the statement is not done yet')
                #raise InternalError('got an empty frame, but the statement is not done yet')

    def _fetch_next_frame(self):
        offset = self._frame.offset + len(self._frame.rows)
        frame = self._connection._client.fetch(
            self._connection._id, self._id,
            offset=offset, frame_max_size=self.itersize)
        self._set_frame(frame)

    def _signature_in_result_set_response(self, result):
        try:
            value = result.signature
            return value
        except:
            return None

    def _first_frame_in_result_set_response(self, result):
        try:
            value = result.first_frame
            return value
        except:
            return None

    def _process_results(self, results):
        if results:
            result = results[0]
            if result.own_statement:
                self._set_id(result.statement_id)
            self._set_signature(result.signature if self._signature_in_result_set_response(result) else None)
            self._set_frame(result.first_frame if self._first_frame_in_result_set_response(result) else None)
            self._updatecount = result.update_count


    def _transform_parameters(self, parameters):
        self._parameter_data_types = []
        parameters_list = list(parameters)
        for idx, param in enumerate(parameters_list):
            if isinstance(param, datetime.datetime):  # only for datetimes
                param = datetimeToTimezoneUtc(param, self._connection.useTZ)
                parameters_list[idx] = param
            typeParam = nativeToParamType(param)
            self._parameter_data_types.append(typeParam)

        typed_parameters = []
        for value, data_type in zip(parameters_list, self._parameter_data_types):

            if value is None:
                typed_parameters.append(TypedValue(null=True, type=Rep.Value('NULL')))
            elif isinstance(value, LXARRAY):

                v = TypedValue()  # parent typed value
                v.type = Rep.Value(data_type[0])
                if value.gettype() is int:
                    v.component_type = Rep.Value("LONG")
                elif value.gettype() is float:
                    v.component_type = Rep.Value("DOUBLE")
                elif value.gettype() is str:
                    v.component_type = Rep.Value("STRING")
                else:
                    raise ArrayTypeException("Array cannot be of type: {}".format(value.gettype()))

                for item in value.getvalue():
                    typeParam = nativeToParamType(item, isArray=True)  # int bigint varchar ???
                    v_temp = TypedValue()
                    v_temp.type = Rep.Value(typeParam[0])
                    if item != None:
                        setattr(v_temp, typeParam[2], typeParam[1](item))
                    else:
                        setattr(v_temp, typeParam[2], 1)  # 1 for True
                    v.array_value.append(v_temp)
                typed_parameters.append(v)
            else:
                v = TypedValue()
                v.type = Rep.Value(data_type[0])
                if data_type[1] is not None:
                    setattr(v, data_type[2], data_type[1](value))
                else:
                    setattr(v, data_type[2], value)

                typed_parameters.append(v)
        return typed_parameters

    def get_tables(self, catalog=None, schemaPattern=None, tableNamePattern=None, typeList=None):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')

        results = self._connection._client.get_tables(self._connection._id, catalog,
                                                      schemaPattern, tableNamePattern, typeList)

        self._process_results([results])

    def get_schemas(self, catalog=None, schemaPattern=None):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        results = self._connection._client.get_schemas(
            self._connection._id, catalog, schemaPattern)

        self._process_results([results])

    def get_catalogs(self):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        results = self._connection._client.get_catalogs(
            self._connection._id)

        self._process_results([results])

    def get_columns(self, catalog=None, schemaPattern=None, tableNamePattern=None, columnNamePattern=None):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')

        results = self._connection._client.get_columns(self._connection._id, catalog,
                                                      schemaPattern, tableNamePattern, columnNamePattern)
        self._process_results([results])

    def get_table_types(self):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        results = self._connection._client.get_table_types(
            self._connection._id)

        self._process_results([results])

    def get_type_info(self):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        results = self._connection._client.get_type_info(
            self._connection._id)

        self._process_results([results])

    def execute(self, operation, parameters=None):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        self._updatecount = -1
        self._set_frame(None)
        if parameters is None or len(parameters) == 0:
            if self._id is None:
                self._set_id(self._connection._client.create_statement(self._connection._id))
            results = self._connection._client.prepare_and_execute(
                self._connection._id, self._id,
                operation, -1, self.itersize)
            self._process_results(results)
        else:
            if operation is not self._prepared_sql:
                statement = self._connection._client.prepare(self._connection._id, operation, -1)
                self._set_id(statement.id)
                self._set_signature(statement.signature)
                self._prepared_sql = operation

            results = self._connection._client.execute(
                self._connection._id, self._id, self._signature, self._transform_parameters(parameters),
                first_frame_max_size=self.itersize)

            self._process_results(results)

    def executemany(self, operation, seq_of_parameters):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        op = operation.split()[0]
        if op.upper() == "SELECT":
            raise ProgrammingError('You cannot execute SELECT statements in executemany()')
        self._updatecount = -1
        self._set_frame(None)
        if operation is not self._prepared_sql:
            statement = self._connection._client.prepare(self._connection._id, operation, max_rows_total=-1)
            self._set_id(statement.id)
            self._set_signature(statement.signature)
            self._prepared_sql = operation
        update_counts = self._connection._client.executeBatch(self._connection._id, self._id, self._signature,
                                         [self._transform_parameters(parameters) for parameters in seq_of_parameters], first_frame_max_size=self.itersize)
        return update_counts

    def executemany2(self, operation, seq_of_parameters):
        if self._closed:
            raise ProgrammingError('the cursor is already closed')
        self._updatecount = -1
        self._set_frame(None)
        statement = self._connection._client.prepare(
            self._connection._id, operation, 0)
        self._set_id(statement.get('id'))
        self._set_signature(statement.get('signature'))
        for parameters in seq_of_parameters:
            self._connection._client.execute(
                statement, self.itersize,
                [parameters])


        '''self._connection._client.execute(
           self._connection._id, self._id,
           statement.signature, self._transform_parameters(parameters),
           first_frame_max_size=0)'''

    def _transform_row(self, row):
        """Transforms a Row into Python values.

        :param row:
            A ``common_pb2.Row`` object.

        :returns:
            A list of values casted into the correct Python types.

        :raises:
            NotImplementedError
        """
        tmp_row = []

        for i, column in enumerate(row.value):
            if column.has_array_value:
                raise NotImplementedError('array types are not supported')
            elif column.scalar_value.null:
                tmp_row.append(None)
            else:
                field_name, rep, mutate_to, cast_from = self._column_data_types[i]

                # get the value from the field_name
                value = getattr(column.scalar_value, field_name)

                # cast the value
                if cast_from is not None:
                    value = cast_from(value)

                tmp_row.append(value)
        return tmp_row

    def fetchone(self):
        result_row = []
        if self._frame is None:
            raise ProgrammingError('no select statement was executed')
        if self._pos is None:
            return None
        rows = self._frame.rows
        row = rows[self._pos]
        self._pos += 1
        if self._pos >= len(rows):
            self._pos = None
            if not self._frame.done:
                self._fetch_next_frame()
        for value, data_type in zip(row.value, self._column_data_types):
            if value.scalar_value.null:
                result_row.append(None)
            else:
                if data_type[1] is not None:
                    val = data_type[1](getattr(value.scalar_value, data_type[2]))
                    if isinstance(val, datetime.datetime):
                        val = timezoneFromDatetimeUtc(val, self._connection.useTZ)
                    result_row.append(val)
                else:
                    result_row.append(getattr(value.scalar_value, data_type[2]))

        return tuple(result_row)

    def fetchone2(self):
        if self._frame is None:
            raise ProgrammingError('no select statement was executed')
        if self._pos is None:
            return None
        rows = self._frame.rows
        row = rows[self._pos]
        #row = self._transform_row(rows[self._pos])
        self._pos += 1
        if self._pos >= len(rows):
            self._pos = None
            if not self._frame.done:
                self._fetch_next_frame()
        return [row]

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize
        rows = []
        while size > 0:
            row = self.fetchone()
            if row is None:
                break
            rows.append(row)
            size -= 1
        return rows

    def fetchall(self):
        rows = []
        while True:
            row = self.fetchone()
            if row is None:
                break
            rows.append(row)
        return rows

    def setinputsizes(self, sizes):
        pass

    def setoutputsize(self, size, column=None):
        pass

    @property
    def connection(self):
        """Read-only attribute providing access to the :class:`Connection <phoenixdb.connection.Connection>`
        object this cursor was created from."""
        return self._connection

    @property
    def rowcount(self):
        """Read-only attribute specifying the number of rows affected by
        the last executed DML statement or -1 if the number cannot be
        determined. Note that this will always be set to -1 for select
        queries."""
        # TODO instead of -1, this ends up being set to Integer.MAX_VALUE
        if self._updatecount == MAX_INT:
            return -1
        return self._updatecount

    @property
    def rownumber(self):
        """Read-only attribute providing the current 0-based index of the
        cursor in the result set or ``None`` if the index cannot be
        determined.

        The index can be seen as index of the cursor in a sequence
        (the result set). The next fetch operation will fetch the
        row indexed by :attr:`rownumber` in that sequence.
        """
        if self._frame is not None and self._pos is not None:
            return self._frame.offset + self._pos
        return self._pos


class DictCursor(Cursor):
    """A cursor which returns results as a dictionary"""

    def _transform_row(self, row):
        row = super(DictCursor, self)._transform_row(row)
        d = {}
        for ind, val in enumerate(row):
            d[self._signature.columns[ind].column_name] = val
        return d
