// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg meas[14];
u2(0,-2.300140836863982) q[0];
u2(0,2.128581626483955) q[1];
cz q[0],q[1];
u2(0,-3.0789369697813296) q[2];
cz q[0],q[2];
cz q[1],q[2];
u2(0,-0.7010097293508428) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
u2(0,-1.459215200412995) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
u2(0,1.7491555959935372) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
u2(0,2.646275250336293) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
u2(0,-2.0026595544002124) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
u2(0,-0.4177283485440526) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
u2(0,1.2983091940603044) q[9];
cz q[0],q[9];
cz q[1],q[9];
cz q[2],q[9];
cz q[3],q[9];
cz q[4],q[9];
cz q[5],q[9];
cz q[6],q[9];
cz q[7],q[9];
cz q[8],q[9];
u2(0,0.5049548463704365) q[10];
cz q[0],q[10];
cz q[1],q[10];
cz q[2],q[10];
cz q[3],q[10];
cz q[4],q[10];
cz q[5],q[10];
cz q[6],q[10];
cz q[7],q[10];
cz q[8],q[10];
cz q[9],q[10];
u2(0,-2.531163330551383) q[11];
cz q[0],q[11];
cz q[1],q[11];
cz q[2],q[11];
cz q[3],q[11];
cz q[4],q[11];
cz q[5],q[11];
cz q[6],q[11];
cz q[7],q[11];
cz q[8],q[11];
cz q[9],q[11];
cz q[10],q[11];
u2(0,2.99568723265606) q[12];
cz q[0],q[12];
cz q[1],q[12];
cz q[2],q[12];
cz q[3],q[12];
cz q[4],q[12];
cz q[5],q[12];
cz q[6],q[12];
cz q[7],q[12];
cz q[8],q[12];
cz q[9],q[12];
cz q[10],q[12];
cz q[11],q[12];
u2(0,2.43439556869545) q[13];
cz q[0],q[13];
u2(0,2.165822076828441) q[0];
cz q[1],q[13];
u2(0,-1.6184206398174066) q[1];
cz q[0],q[1];
cz q[2],q[13];
u2(0,-2.861076992562003) q[2];
cz q[0],q[2];
cz q[1],q[2];
cz q[3],q[13];
u2(0,-1.7382968496642348) q[3];
cz q[0],q[3];
cz q[1],q[3];
cz q[2],q[3];
cz q[4],q[13];
u2(0,-1.0004146447240423) q[4];
cz q[0],q[4];
cz q[1],q[4];
cz q[2],q[4];
cz q[3],q[4];
cz q[5],q[13];
u2(0,-2.5885433378940865) q[5];
cz q[0],q[5];
cz q[1],q[5];
cz q[2],q[5];
cz q[3],q[5];
cz q[4],q[5];
cz q[6],q[13];
u2(0,2.614028960054684) q[6];
cz q[0],q[6];
cz q[1],q[6];
cz q[2],q[6];
cz q[3],q[6];
cz q[4],q[6];
cz q[5],q[6];
cz q[7],q[13];
u2(0,-0.14555942532043709) q[7];
cz q[0],q[7];
cz q[1],q[7];
cz q[2],q[7];
cz q[3],q[7];
cz q[4],q[7];
cz q[5],q[7];
cz q[6],q[7];
cz q[8],q[13];
u2(0,1.4491343898378215) q[8];
cz q[0],q[8];
cz q[1],q[8];
cz q[2],q[8];
cz q[3],q[8];
cz q[4],q[8];
cz q[5],q[8];
cz q[6],q[8];
cz q[7],q[8];
cz q[9],q[13];
u2(0,-0.25247180722801055) q[9];
cz q[0],q[9];
cz q[1],q[9];
cz q[2],q[9];
cz q[3],q[9];
cz q[4],q[9];
cz q[5],q[9];
cz q[6],q[9];
cz q[7],q[9];
cz q[8],q[9];
cz q[10],q[13];
u2(0,2.832613520498075) q[10];
cz q[0],q[10];
cz q[1],q[10];
cz q[2],q[10];
cz q[3],q[10];
cz q[4],q[10];
cz q[5],q[10];
cz q[6],q[10];
cz q[7],q[10];
cz q[8],q[10];
cz q[9],q[10];
cz q[11],q[13];
u2(0,-0.1827160341009062) q[11];
cz q[0],q[11];
cz q[1],q[11];
cz q[2],q[11];
cz q[3],q[11];
cz q[4],q[11];
cz q[5],q[11];
cz q[6],q[11];
cz q[7],q[11];
cz q[8],q[11];
cz q[9],q[11];
cz q[10],q[11];
cz q[12],q[13];
u2(0,2.180074668076413) q[12];
cz q[0],q[12];
cz q[1],q[12];
cz q[2],q[12];
cz q[3],q[12];
cz q[4],q[12];
cz q[5],q[12];
cz q[6],q[12];
cz q[7],q[12];
cz q[8],q[12];
cz q[9],q[12];
cz q[10],q[12];
cz q[11],q[12];
u2(0,-1.293618951366648) q[13];
cz q[0],q[13];
u2(0,-0.2741796505906633) q[0];
cz q[1],q[13];
u2(0,1.5147502908811727) q[1];
cz q[2],q[13];
u2(0,-2.3485457400850827) q[2];
cz q[3],q[13];
u2(0,0.22134274439856227) q[3];
cz q[4],q[13];
u2(0,0.2585364894485487) q[4];
cz q[5],q[13];
u2(0,-0.41709129527490774) q[5];
cz q[6],q[13];
u2(0,2.2305844777618233) q[6];
cz q[7],q[13];
u2(0,-1.4980262732238527) q[7];
cz q[8],q[13];
u2(0,1.2131989240211833) q[8];
cz q[9],q[13];
u2(0,0.4668970936654304) q[9];
cz q[10],q[13];
u2(0,-0.1755787753733955) q[10];
cz q[11],q[13];
u2(0,-2.6900410043905447) q[11];
cz q[12],q[13];
u2(0,-2.61076170857186) q[12];
u2(0,2.4738510991662945) q[13];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],q[12],q[13];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];
measure q[12] -> meas[12];
measure q[13] -> meas[13];