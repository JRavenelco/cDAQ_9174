/*
 * CBR_KAN_Calibrated.h
 * Casos de Oro CALIBRADOS con datos experimentales
 * Generado automáticamente por calibrar_cbr_kan.py
 */

#ifndef CBR_KAN_CALIBRATED_H
#define CBR_KAN_CALIBRATED_H

// Casos de Oro calibrados experimentalmente
// Formato: {y[m], i_eq[A], u_eq[V], phi[Wb], L[H], di/dt+, di/dt-, kp, ki, conf, n}
static GoldenCase golden_cases_calibrated[NUM_GOLDEN_CASES] = {
    {0.002f, 0.230f, 4.51f, 0.0215f, 0.0937f, 105.2f, -94.7f, 350.0f, 80.0f, 0.68f, 68},
    {0.004f, 0.422f, 6.71f, 0.0369f, 0.0875f, 112.7f, -101.4f, 340.0f, 75.0f, 1.00f, 642},
    {0.006f, 0.500f, 7.32f, 0.0418f, 0.0835f, 118.0f, -106.2f, 330.0f, 70.0f, 1.00f, 443},
    {0.008f, 0.493f, 6.96f, 0.0399f, 0.0808f, 122.1f, -109.9f, 320.0f, 65.0f, 1.00f, 137},
    {0.010f, 0.496f, 6.81f, 0.0391f, 0.0788f, 125.2f, -112.7f, 310.0f, 60.0f, 0.76f, 76},
    {0.012f, 0.549f, 7.37f, 0.0424f, 0.0772f, 127.7f, -114.9f, 300.0f, 55.0f, 0.58f, 58},
    {0.014f, 0.585f, 7.58f, 0.0445f, 0.0760f, 129.7f, -116.7f, 290.0f, 50.0f, 0.68f, 68},
    {0.016f, 0.630f, 1.39f, 0.0473f, 0.0750f, 131.4f, -118.3f, 280.0f, 45.0f, 0.50f, 0},
    {0.018f, 0.690f, 1.52f, 0.0512f, 0.0742f, 132.9f, -119.6f, 270.0f, 40.0f, 0.50f, 0},
    {0.020f, 0.750f, 1.65f, 0.0552f, 0.0735f, 134.1f, -120.7f, 260.0f, 35.0f, 0.50f, 0},
};

#endif // CBR_KAN_CALIBRATED_H
