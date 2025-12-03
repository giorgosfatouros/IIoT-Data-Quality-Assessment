-- ============================================================================
-- Populate Sensor Thresholds from tags.csv
-- ============================================================================
-- This script migrates sensor metadata from tags.csv to sensor_thresholds table
-- Data is inserted with proper handling of NULL values

-- Clear existing data (optional - comment out if you want to keep existing data)
-- TRUNCATE TABLE sensor_thresholds;

-- Insert sensor thresholds
-- Note: Empty strings in CSV are treated as NULL
INSERT INTO sensor_thresholds (tag, tag_description, machine_group, low_threshold, high_threshold, threshold_type, aggregation_rule, engineering_units, category)
VALUES
('22PI102', 'SEAL OIL MAIN PUMP PRESSURE', 'K-2201_KT-2201', 6, NULL, 'Down', 'min', 'Kgf/cm2', 'Pressure'),
('22PI103', 'CONTROL OIL HEADER PRESSURE', 'K-2201_KT-2201', 5, NULL, 'Down', 'min', 'Kgf/cm2', 'Pressure'),
('22PI69', 'L.O MAIN PUMP DELIVERY LOW PRESSURE', 'K-2201_KT-2201', 7.1, NULL, 'Down', 'min', 'Kgf/cm2', 'Pressure'),
('22PI70', 'LUBE OIL HEADER PRESSURE', 'K-2201_KT-2201', 6.6, NULL, 'Down', 'min', 'Kgf/cm2', 'Pressure'),
('22PI95', 'L.O. HEADER PRESSURE', 'K-2201_KT-2201', 0.96, NULL, 'Down', 'min', 'Kgf/cm2', 'Pressure'),
('22SI101', 'Shutdown indicator when value <100', 'K-2201_KT-2201', NULL, NULL, NULL, 'avg', 'rpm', NULL),
('22TI111', 'ST. TURBINE N.D.E BEARING TEMPERATURE', 'K-2201_KT-2201', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('22TI113', 'ST. TURBINE D.E BEARING TEMPERATURE', 'K-2201_KT-2201', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('22TI115', 'D.E BEARING TEMPERATURE', 'K-2201_KT-2201', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('22TI117', 'N.D.E BEARING TEMPERATURE', 'K-2201_KT-2201', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('22TI123', 'ST. TURBINE BEARING ACT. SIDE TEMPERATURE', 'K-2201_KT-2201', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('22VI01', 'ST. TURBINE SHAFT RADIAL VIBRATION', 'K-2201_KT-2201', NULL, 80, 'Up', 'max', 'micron', 'Vibration'),
('22VI04', 'ST. TURBINE SHAFT RADIAL VIBRATION', 'K-2201_KT-2201', NULL, 80, 'Up', 'max', 'micron', 'Vibration'),
('22VI06', 'SHAFT RADIAL VIBRATION', 'K-2201_KT-2201', NULL, 115, 'Up', 'max', 'micron', 'Vibration'),
('22VI08', 'SHAFT RADIAL VIBRATION', 'K-2201_KT-2201', NULL, 110, 'Up', 'max', 'micron', 'Vibration'),
('22ZI09', 'ST. TURBINE SHAFT AXIAL DISPLACEMENT', 'K-2201_KT-2201', -0.4000000059604645, 0.4, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('22ZI10', 'SHAFT AXIAL DISPLACEMENT', 'K-2201_KT-2201', -0.4000000059604645, 0.4000000059604645, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('22ZI11', 'ST. TURBINE SHAFT AXIAL DISPLACEMENT', 'K-2201_KT-2201', -0.4000000059604645, 0.4, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('32JI001A', 'MOTOR CURRENT', 'K-3201 A_KM-3201 A', NULL, 725, 'Up', 'max', 'Ampere', 'Motor Current'),
('32PI424', 'LUBE OIL INLET PRESSURE ', 'K-3201 A_KM-3201 A', 1.5, NULL, 'Down', 'min', 'kg/cm2', 'Pressure'),
('32TI448', 'MOTOR STATOR TEMPERATURE ', 'K-3201 A_KM-3201 A', NULL, 110, 'Up', 'max', 'degC', 'Temperature'),
('32TI448_2', 'MOTOR STATOR TEMPERATURE', 'K-3201 A_KM-3201 A', NULL, 105, 'Up', 'max', 'degC', 'Temperature'),
('32TI448_4', 'MOTOR STATOR TEMPERATURE', 'K-3201 A_KM-3201 A', NULL, 105, 'Up', 'max', 'degC', 'Temperature'),
('32TI448_6', 'MOTOR STATOR TEMPERATURE', 'K-3201 A_KM-3201 A', NULL, 105, 'Up', 'max', 'degC', 'Temperature'),
('32TI448_7', 'MOTOR  NDE  BEARING TEMPERATURE ', 'K-3201 A_KM-3201 A', NULL, 80, 'Up', 'max', 'degC', 'Temperature'),
('32TI448_8', 'MOTOR  DE  BEARING TEMPERATURE ', 'K-3201 A_KM-3201 A', NULL, 80, 'Up', 'max', 'degC', 'Temperature'),
('32XI451', 'GEARBOX JOURNAL BEARING VIBRATION (PINION )', 'K-3201 A_KM-3201 A', NULL, 75, 'Up', 'max', 'micron', 'Vibration'),
('32XI452', 'GEARBOX JOURNAL BEARING VIBRATION (PINION )', 'K-3201 A_KM-3201 A', NULL, 50, 'Up', 'max', 'micron', 'Vibration'),
('32XI453', 'GEARBOX AXIAL DISPLACEMENT (PINION )', 'K-3201 A_KM-3201 A', -0.300000011920929, 0.300000011920929, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('32XI454', 'GEARBOX AXIAL DISPLACEMENT (GEAR )', 'K-3201 A_KM-3201 A', -0.300000011920929, 0.300000011920929, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('32XI455', 'VIBRATION JOURNAL BEARING (DISCHARGE END-JOURNAL )', 'K-3201 A_KM-3201 A', NULL, 70, 'Up', 'max', 'micron', 'Vibration'),
('32XI456', 'VIBRATION JOURNAL BEARING  (INTAKE END-THRUST /JOURNAL )', 'K-3201 A_KM-3201 A', NULL, 70, 'Up', 'max', 'micron', 'Vibration'),
('32XI457', 'AXIAL DISPLACEMENT COMPRESSOR (INTAKE END)', 'K-3201 A_KM-3201 A', -0.449999988079071, 0.449999988079071, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('33AI601', 'AXIAL DISPLACEMENT TURBINE (EXHAUST END)', 'K-3301 B_KT-3301 B', -0.300000011920929, 0.3499999940395355, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('33AI602', 'AXIAL DISPLACEMENT COMPRESSOR (INTAKE END)', 'K-3301 B_KT-3301 B', -0.300000011920929, 0.300000011920929, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('33PI222', 'LUBE OIL INLET PRESSURE ', 'K-3301 B_KT-3301 B', 1.5, NULL, 'Down', 'min', 'kg/cm2', 'Pressure'),
('33PI601', 'CONTROL OIL OUTLET PRESSURE', 'K-3301 B_KT-3301 B', 9, NULL, 'Down', 'min', 'kg/cm2', 'Pressure'),
('33SI501A', 'Shutdown indicator when value <100', 'K-3301 B_KT-3301 B', NULL, 10000, 'Up', 'max', 'rpm', NULL),
('33TI602', 'JOURNAL BEARING TEMPERATURE (STEAM END)', 'K-3301 B_KT-3301 B', NULL, 100, 'Up', 'max', 'degC', 'Temperature'),
('33TI603', 'JOURNAL BEARING TEMPERATURE (EXHAUST END)', 'K-3301 B_KT-3301 B', NULL, 100, 'Up', 'max', 'degC', 'Temperature'),
('33TI604', 'THRUST PADS OUTBOARD TEMPERATURE TURBINE', 'K-3301 B_KT-3301 B', NULL, 125, 'Up', 'max', 'degC', 'Temperature'),
('33TI606', 'THRUST PADS INBOARD TEMPERATURE TURBINE ', 'K-3301 B_KT-3301 B', NULL, 125, 'Up', 'max', 'degC', 'Temperature'),
('33TI607', 'JOURNAL PADS TEMPERATURE (DISCHARGE END)', 'K-3301 B_KT-3301 B', NULL, 125, 'Up', 'max', 'degC', 'Temperature'),
('33TI608', 'JOURNAL PADS TEMPERATURE (INTAKE END)', 'K-3301 B_KT-3301 B', NULL, 125, 'Up', 'max', 'degC', 'Temperature'),
('33TI609', 'THRUST PADS INBOARD TEMPERATURE', 'K-3301 B_KT-3301 B', NULL, 125, 'Up', 'max', 'degC', 'Temperature'),
('33TI611', 'THRUST PADS OUTBOARD TEMPERATURE', 'K-3301 B_KT-3301 B', NULL, 125, 'Up', 'max', 'degC', 'Temperature'),
('33TI613', 'THRUST PADS INBOARD TEMPERATURE TURBINE ', 'K-3301 B_KT-3301 B', NULL, 125, 'Up', 'max', 'degC', 'Temperature'),
('33VI601', 'VIBRATION JOURNAL BEARING (STEAM END- JOURNAL)', 'K-3301 B_KT-3301 B', NULL, 50, 'Up', 'max', 'micron', 'Vibration'),
('33VI602', 'VIBRATION JOURNAL BEARING TURBINE (EXHAUST END- THRUST/JOURNAL)', 'K-3301 B_KT-3301 B', NULL, 82, 'Up', 'max', 'micron', 'Vibration'),
('33VI603', 'VIBRATION JOURNAL BEARING (DISCHARGE END- JOURNAL- COMPRESSOR', 'K-3301 B_KT-3301 B', NULL, 75, 'Up', 'max', 'micron', 'Vibration'),
('33VI604', 'VIBRATION JOURNAL BEARING (INTAKE END- THRUST/JOURNAL -COMPRESSOR)', 'K-3301 B_KT-3301 B', NULL, 75, 'Up', 'max', 'micron', 'Vibration'),
('57FI001A', 'C. AIR DISCHARGE FLOW ', 'K-5701', 1800, NULL, 'Down', 'min', 'Kg/h', 'Flow'),
('57PI005', 'C. AIR DISCHARGE PRESSURE', 'K-5701', 0.5, 0.6000000238418579, 'Up/Down', 'avg', 'Kg/cm2', 'Pressure'),
('57PI014', 'GEAR BOX LUBE-OIL PRESSURE', 'K-5701', 1, NULL, 'Down', 'min', 'Kg/cm2', 'Pressure'),
('57PIC004', 'C.AIR BLOW OFF PRESSURE', 'K-5701', 0.5, NULL, 'Down', 'min', 'Kg/cm2', 'Pressure'),
('57TI003', 'C.AIR DISCHARGE TEMPERATURE ', 'K-5701', NULL, 115, 'Up', 'max', 'degC', 'Temperature'),
('57TI030', 'GEAR BOX TEMPERATURE', 'K-5701', NULL, 90, 'Up', 'max', 'degC', 'Temperature'),
('75PDI853', 'PRIMARY VENT DIFFERENTIAL PRESSURE', 'K-7502_ST-7501', NULL, 3, 'Up', 'max', 'kg/cm2', 'Pressure'),
('75PI808', 'LUBE OIL MAIN PUMP DELIVERY PRESSURE', 'K-7502_ST-7501', NULL, NULL, NULL, 'avg', 'kg/cm2', 'Pressure'),
('75PI823', 'LUBE OIL HEADER PRESSURE', 'K-7502_ST-7501', 1.799999952316280, NULL, 'Down', 'min', 'kg/cm2', 'Pressure'),
('75PI845', 'N2 INLET PRESSURE', 'K-7502_ST-7501', 0.84, NULL, 'Down', 'min', 'kg/cm2', 'Pressure'),
('75PI870', 'CONTROL OIL HEADER PRESSURE ', 'K-7502_ST-7501', 4.2, NULL, 'Down', 'min', 'kg/cm2', 'Pressure'),
('75SI865R', 'Shutdown indicator when value <100', 'K-7502_ST-7501', 6050, 8500, 'Up/Down', 'avg', 'rpm', NULL),
('75TI821', 'STEAM TURBINE NG 40/32 THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI822', 'STEAM TURBINE NG 40/32 THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI824', 'STEAM TURBINE NG 40/32 JOURNAL BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI827', 'STEAM TURBINE NG 40/32 THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI828', 'STEAM TURBINE NG 40/32 THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI829', 'COMPRESSOR BCL 509/A  THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI830', 'COMPRESSOR BCL 509/A  THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI831', 'COMPRESSOR BCL 509/A  THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI832', 'COMPRESSOR BCL 509/A  THRUST BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI834', 'COMPRESSOR BCL 509/A  JOURNAL BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75TI836', 'COMPRESSOR BCL 509/A  JOURNAL BEARING TEMPERATURE', 'K-7502_ST-7501', NULL, 120, 'Up', 'max', 'degC', 'Temperature'),
('75XI821BX', 'STEAM TURBINE NG 40/32 JOURNAL BEARING RAD. VIBRATION', 'K-7502_ST-7501', NULL, 89, 'Up', 'max', 'micron', 'Vibration'),
('75XI822BY', 'STEAM TURBINE NG 40/32 JOURNAL BEARING RAD. VIBRATION', 'K-7502_ST-7501', NULL, 89, 'Up', 'max', 'micron', 'Vibration'),
('75XI823BX', 'COMPRESSOR BCL 509/A  JOURNAL BEARING RAD. VIBRATION', 'K-7502_ST-7501', NULL, 89, 'Up', 'max', 'micron', 'Vibration'),
('75XI824BX', 'COMPRESSOR BCL 509/A  JOURNAL BEARING RAD. VIBRATION', 'K-7502_ST-7501', NULL, 89, 'Up', 'max', 'micron', 'Vibration'),
('75ZI800BA', 'STEAM TURBINE NG 40/32 SHAFT AXIAL DISPLACEMENT', 'K-7502_ST-7501', -0.5, 0.5, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('75ZI800BB', 'STEAM TURBINE NG 40/32 SHAFT AXIAL DISPLACEMENT', 'K-7502_ST-7501', -0.5, 0.5, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('75ZI801BA', 'COMPRESSOR BCL 509/A SHAFT AXIAL DISPLACEMENT', 'K-7502_ST-7501', -0.5, 0.5, 'Up/Down', 'avg', 'mm', 'Axial Displacement'),
('75ZI801BB', 'COMPRESSOR BCL 509/A SHAFT AXIAL DISPLACEMENT', 'K-7502_ST-7501', -0.5, 0.5, 'Up/Down', 'avg', 'mm', 'Axial Displacement')
ON CONFLICT (tag) DO UPDATE SET
    tag_description = EXCLUDED.tag_description,
    machine_group = EXCLUDED.machine_group,
    low_threshold = EXCLUDED.low_threshold,
    high_threshold = EXCLUDED.high_threshold,
    threshold_type = EXCLUDED.threshold_type,
    aggregation_rule = EXCLUDED.aggregation_rule,
    engineering_units = EXCLUDED.engineering_units,
    category = EXCLUDED.category,
    updated_at = NOW();

-- Verify data was inserted
SELECT COUNT(*) as total_sensors, COUNT(DISTINCT machine_group) as machine_groups FROM sensor_thresholds;




