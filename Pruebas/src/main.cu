#include <iostream>
#include <algorithm>         // Para std::min
#include <NIDAQmx.h>
#include <cuda_runtime.h>

// Tamaño del buffer para la adquisición
#define BUFFER_SIZE 1024

// Kernel CUDA de ejemplo: eleva cada elemento al cuadrado
__global__ void processData(float* d_data, int numSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        d_data[idx] = d_data[idx] * d_data[idx];
    }
}

int main() {
    // --- Sección 1: Adquisición de datos con NI-DAQmx ---
    TaskHandle taskHandle = 0;
    float64 data[BUFFER_SIZE] = {0};  // NI-DAQmx usa double (float64)
    int32 samplesRead = 0;
    
    // Crear tarea
    if (DAQmxCreateTask("", &taskHandle) != 0) {
        std::cerr << "Error al crear la tarea." << std::endl;
        return -1;
    }
    
    // Crear canal de entrada analógica (ajusta el nombre del canal según tu configuración)
    if (DAQmxCreateAIVoltageChan(taskHandle, "cDAQ1Mod1/ai0", "", DAQmx_Val_Cfg_Default,
                                 -10.0, 10.0, DAQmx_Val_Volts, NULL) != 0) {
        std::cerr << "Error al crear el canal de entrada analógica." << std::endl;
        DAQmxClearTask(taskHandle);
        return -1;
    }
    
    // Iniciar la tarea
    if (DAQmxStartTask(taskHandle) != 0) {
        std::cerr << "Error al iniciar la tarea." << std::endl;
        DAQmxClearTask(taskHandle);
        return -1;
    }
    
    // Leer datos: adquiere BUFFER_SIZE muestras con un timeout de 10 segundos
    if (DAQmxReadAnalogF64(taskHandle, BUFFER_SIZE, 10.0, DAQmx_Val_GroupByScanNumber,
                           data, BUFFER_SIZE, &samplesRead, NULL) != 0) {
        std::cerr << "Error al leer datos." << std::endl;
        DAQmxStopTask(taskHandle);
        DAQmxClearTask(taskHandle);
        return -1;
    }
    
    // Finalizar la tarea
    DAQmxStopTask(taskHandle);
    DAQmxClearTask(taskHandle);
    
    std::cout << "Se adquirieron " << samplesRead << " muestras." << std::endl;
    
    // (Opcional) Mostrar algunos datos adquiridos
    // Usamos std::min<int> para evitar error de sobrecarga con int32 e int
    for (int i = 0; i < std::min<int>(samplesRead, 10); i++) {
        std::cout << "Dato[" << i << "]: " << data[i] << " V" << std::endl;
    }
    
    // --- Sección 2: Procesamiento con CUDA ---
    // Convertir los datos de double a float para procesarlos en la GPU
    float h_data[BUFFER_SIZE];
    for (int i = 0; i < samplesRead; i++) {
        h_data[i] = static_cast<float>(data[i]);
    }
    
    float* d_data = nullptr;
    cudaError_t cudaStatus;
    
    // Reservar memoria en la GPU para los datos
    cudaStatus = cudaMalloc((void**)&d_data, samplesRead * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error al reservar memoria en la GPU: " << cudaGetErrorString(cudaStatus) << std::endl;
        return -1;
    }
    
    // Copiar los datos desde la CPU a la GPU
    cudaStatus = cudaMemcpy(d_data, h_data, samplesRead * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error al copiar datos a la GPU: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_data);
        return -1;
    }
    
    // Configurar y lanzar el kernel CUDA
    int threadsPerBlock = 256;
    int blocksPerGrid = (samplesRead + threadsPerBlock - 1) / threadsPerBlock;
    processData<<<blocksPerGrid, threadsPerBlock>>>(d_data, samplesRead);
    
    // Sincronizar y verificar errores en el kernel
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error en la ejecución del kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_data);
        return -1;
    }
    
    // Copiar los datos procesados de regreso a la CPU
    cudaStatus = cudaMemcpy(h_data, d_data, samplesRead * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error al copiar datos desde la GPU: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_data);
        return -1;
    }
    
    // Liberar la memoria en la GPU
    cudaFree(d_data);
    
    // Mostrar algunos de los datos procesados
    std::cout << "\nDatos procesados (elevados al cuadrado):" << std::endl;
    for (int i = 0; i < std::min<int>(samplesRead, 10); i++) {
        std::cout << "Dato[" << i << "]: " << h_data[i] << std::endl;
    }
    
    return 0;
}
