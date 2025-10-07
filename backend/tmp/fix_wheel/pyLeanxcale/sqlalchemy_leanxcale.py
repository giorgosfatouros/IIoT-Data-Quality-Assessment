# Copyright 2017 Dimitri Capitaine
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
from future.standard_library import install_aliases
install_aliases()

import urllib.parse
import pyLeanxcale

from sqlalchemy.engine.default import DefaultDialect

from sqlalchemy.sql.compiler import DDLCompiler, SQLCompiler
from sqlalchemy.sql import compiler
from sqlalchemy.engine import default
from sqlalchemy.exc import CompileError

from sqlalchemy import types, BigInteger, Float
from sqlalchemy.types import INTEGER, BIGINT, SMALLINT, VARCHAR, CHAR, \
    FLOAT, DATE, BOOLEAN, DECIMAL, TIMESTAMP, TIME, VARBINARY


class LeanxcaleCompiler(SQLCompiler):
    def visit_sequence(self, seq):
        return "NEXT VALUE FOR %s" % seq.name

class LeanxcaleDDLCompiler(DDLCompiler):

    def visit_primary_key_constraint(self, constraint):
        # if constraint.name is None:
        #    raise CompileError("can't create primary key without a name")
        return DDLCompiler.visit_primary_key_constraint(self, constraint)

    def visit_sequence(self, seq):
        return "NEXT VALUE FOR %s" % seq.name

class LeanxcaleExecutionContext(default.DefaultExecutionContext):

    def should_autocommit_text(self, statement):
        pass

    def create_server_side_cursor(self):
        pass

    def fire_sequence(self, seq, type_):
        return self._execute_scalar(
            (
                    "SELECT NEXT VALUE FOR %s"
                    % seq.name
            ),
            type_
        )


class LeanXcaleIdentifierPreparer(compiler.IdentifierPreparer):

    def __init__(self, dialect, server_ansiquotes=False, **kw):

        quote = '"'

        super(LeanXcaleIdentifierPreparer, self).__init__(
            dialect, initial_quote=quote, escape_quote=quote
        )

    def _quote_free_identifiers(self, *ids):
        """Unilaterally identifier-quote any number of strings."""

        return tuple([self.quote_identifier(i) for i in ids if i is not None])

class LeanXcaleGenericTypeCompiler(compiler.GenericTypeCompiler):
    def visit_ARRAY(self, type_, **kw):
        if type_.item_type.python_type == int:
            return "BIGINT ARRAY"
        elif type_.item_type.python_type == float:
            return "DOUBLE ARRAY"
        elif type_.item_type.python_type == str:
            return "VARCHAR ARRAY"
        else:
            raise Exception("ARRAY of type {} is not supported".format(str(type_)))


class LeanxcaleDialect(DefaultDialect):
    name = "leanxcale"

    driver = "pyleanxcale"

    ddl_compiler = LeanxcaleDDLCompiler
    preparer = LeanXcaleIdentifierPreparer
    statement_compiler = LeanxcaleCompiler
    type_compiler = LeanXcaleGenericTypeCompiler

    supports_sequences = True
    sequences_optional = True
    supports_multivalues_insert = True

    execution_ctx_cls = LeanxcaleExecutionContext

    preexecute_autoincrement_sequences = True

    @classmethod
    def dbapi(cls):

        return pyLeanxcale

    def create_connect_args(self, url):

        # Deal with PARALLEL option
        parallel = url.query.get('parallel')

        # Deal with SECURE option
        secure = url.query.get('secure')

        leanxcale_url = urllib.parse.urlunsplit(urllib.parse.SplitResult(
            scheme='http',
            netloc='{}:{}'.format(url.host, url.port or 8765),
            path='/',
            query=urllib.parse.urlencode(url.query),
            fragment='',
        ))
        if (url.query.get('autocommit') == 'False'):
            autocommit = False
        else:
            autocommit = True
        # By default, do not use timezoned timestamps
        if (url.query.get('timezonedTimestamps') == 'True'):
            timezone = True
        else:
            timezone = False

        params = {'autocommit': autocommit, 'user': url.username}
        if parallel is not None:
            params.update({'parallel': parallel})
        if secure is not None:
            params.update({'secure': secure})
        if url.password is not None:
            params.update({'password': url.password})
        params.update({'useTZ': timezone})

        params.update({'databaseName': url.database})
        return [leanxcale_url], params

    def _get_default_schema_name(self, connection):
        return connection.engine.url.username.upper()

    def do_rollback(self, dbapi_conection):
        dbapi_conection.rollback()

    def do_commit(self, dbapi_conection):
        dbapi_conection.commit()

    def has_sequence(self, connection, sequence_name, schema=None, **kw):
        return sequence_name.upper() in self.get_sequence_names(connection, schema)

    def get_sequence_names(self, connection, schema=None, **kw):
        dbapi_con = connection.connect().connection
        cursor = dbapi_con.cursor()
        typeList = ["SEQUENCE"]
        schema = schema.upper() if schema is not None else self.default_schema_name
        cursor.get_tables(schemaPattern=schema, typeList=typeList)
        sequences = cursor.fetchall()

        result = []
        for seq in sequences:
            result.append(seq[2])
        return result

    def has_table(self, connection, table_name, schema=None, **kw):
        return table_name.upper() in self.get_table_names(connection, schema)

    def get_schema_names(self, connection, catalog=None, schema=None, **kw):
        dbapi_con = connection.connect().connection
        cursor = dbapi_con.cursor()
        schema = schema.upper() if schema is not None else self.default_schema_name
        cursor.get_schemas(catalog, schema)
        schemas = cursor.fetchall()

        result = []
        for schema in schemas:
            result.append(schema[0])
        return result

    def get_table_names(self, connection, schema=None, **kw):

        dbapi_con = connection.connect().connection
        cursor = dbapi_con.cursor()
        tableNamePattern = None if 'tableNamePattern' not in kw.keys() else kw.get('tableNamePattern')
        typeList = ['TABLE', 'MATERIALIZED QUERY TABLE'] if 'typeList' not in kw.keys() else kw.get('typeList')
        schema = schema.upper() if schema is not None else self.default_schema_name
        cursor.get_tables(schemaPattern=schema, tableNamePattern=tableNamePattern, typeList=typeList)
        tables = cursor.fetchall()
        result = []
        for table in tables:
            result.append(table[2])
        return result

    def get_columns(self, connection, table_name=None, schema=None, catalog=None, column_name=None, **kw):

        dbapi_con = connection.connect().connection
        cursor = dbapi_con.cursor()
        schema = schema.upper() if schema is not None else self.default_schema_name
        cursor.get_columns(catalog, schema, table_name, column_name)
        columns = cursor.fetchall()

        result = []
        for column in columns:
            col_d = {}
            col_d.update({'name': column[3]})
            if column[4] == 2003:
                if column[5].split()[0] == 'BIGINT':
                    col_d.update({'item_type': BigInteger})
                elif column[5].split()[0] == 'DOUBLE':
                    col_d.update({'item_type': Float})
                elif column[5].split()[0] == 'VARCHAR':
                    col_d.update({'item_type': VARCHAR})
                col_d.update({'type': ARRAY(column[5])})
            else:
                col_d.update({'type': COLUMN_DATA_TYPE[column[4]]})
            col_d.update({'nullable': column[10] == 1 if True else False})
            col_d.update({'default': column[13]})

            result.append(col_d)

        return result

    def get_view_names(self, connection, schema=None, **kw):
        dbapi_con = connection.connect().connection
        cursor = dbapi_con.cursor()
        schema = schema.upper() if schema is not None else self.default_schema_name
        cursor.get_tables(schemaPattern=schema, typeList=['VIEW'])
        tables = cursor.fetchall()

        result = []
        for table in tables:
            result.append(table[2])
        return result

    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        pk_sql = "SELECT * FROM LXSYSMETA.PRIMARY_KEYS " \
                 "WHERE TABLENAME = '" + table_name +"' ORDER BY keyseq"
        dbapi_con = connection.connect()
        rs = dbapi_con.execute(pk_sql)
        pk_list = [r for r in rs]
        print("pk_list: {}".format(pk_list))
        result = {
            'constrained_columns': [],
        }
        [result['constrained_columns'].append(pk[0]) for pk in pk_list]
        return result

    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        return []

    def get_indexes(self, connection, table_name, schema=None, **kw):
        index_sql = "SELECT * FROM LXSYSMETA.INDEXES WHERE TABLENAME = '" + table_name +"' order by indexname, ordinalposition"
        dbapi_con = connection.connect()
        rs = dbapi_con.execute(index_sql)
        index_list = [r for r in rs]

        result = []

        for index in index_list:
            name = index[2]
            column = index[5]
            nonUnique = index[0]

            if len(result) == 0:
                result.append({
                    'name': name,
                    'column_names': [column],
                    'unique': not nonUnique
                })
            else:
                found = False
                for final_index in result:
                    if name == final_index.get('name'):
                        found = True
                        final_index['column_names'].append(column)
                if not found:
                    result.append({
                        'name': name,
                        'column_names': [column],
                        'unique': not nonUnique
                    })
        return result

    #def has_index(self, connection, table_name, index_name, schema=None):
    #    pass

class TINYINT(types.Integer):
    __visit_name__ = "INTEGER"


class UTINYINT(types.Integer):
    __visit_name__ = "INTEGER"


class UINTEGER(types.Integer):
    __visit_name__ = "INTEGER"


class DOUBLE(types.BIGINT):
    __visit_name__ = "BIGINT"


class DOUBLE(types.BIGINT):
    __visit_name__ = "BIGINT"


class UDOUBLE(types.BIGINT):
    __visit_name__ = "BIGINT"


class UFLOAT(types.FLOAT):
    __visit_name__ = "FLOAT"


class ULONG(types.BIGINT):
    __visit_name__ = "BIGINT"


class UTIME(types.TIME):
    __visit_name__ = "TIME"


class UDATE(types.DATE):
    __visit_name__ = "DATE"


class UTIMESTAMP(types.TIMESTAMP):
    __visit_name__ = "TIMESTAMP"


class ROWID(types.String):
    __visit_name__ = "VARCHAR"

class ARRAY(types.ARRAY):
    __visit_name__ = "ARRAY"
    def __init__(self, type):
        types.ARRAY.__init__(self, type)



COLUMN_DATA_TYPE = {
    -6: TINYINT,
    -5: BIGINT,
    -3: VARBINARY,
    1: CHAR,
    3: DECIMAL,
    4: INTEGER,
    5: SMALLINT,
    6: FLOAT,
    8: DOUBLE,
    9: UINTEGER,
    10: ULONG,
    11: UTINYINT,
    12: VARCHAR,
    13: ROWID,
    14: UFLOAT,
    15: UDOUBLE,
    16: BOOLEAN,
    18: UTIME,
    19: UDATE,
    20: UTIMESTAMP,
    91: DATE,
    92: TIME,
    93: TIMESTAMP
}
