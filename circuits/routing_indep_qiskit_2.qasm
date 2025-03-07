// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg meas[2];
ry(-0.2977358082261536) q[0];
ry(1.1765602240048205) q[1];
cx q[0],q[1];
ry(0.032455418298796515) q[0];
ry(-2.571400467166911) q[1];
cx q[0],q[1];
ry(-0.34169442553741997) q[0];
ry(2.082032939699599) q[1];
cx q[0],q[1];
ry(-2.9458911158390353) q[0];
ry(2.478962354026642) q[1];
barrier q[0],q[1];
measure q[0] -> meas[0];
measure q[1] -> meas[1];