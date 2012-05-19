/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This example demonstrates how to use the Cuda OpenGL bindings with the
 * runtime API.
 * Device code.
 */

#ifndef _SIMPLEGL_KERNEL_H_
#define _SIMPLEGL_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef unsigned char Pixel;

// Texture reference for reading image
texture<unsigned char, 2> tex;
static cudaArray *array = NULL;

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    //int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned char pix00 = tex2D(tex, (float) x, (float) y);

    // calculate simple sine wave pattern
    // float freq = 4.0f;
    // float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // The vertex's height will be the grayscale value of the image
    float w = (float)pix00 / (float)UCHAR_MAX; //sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

    // write output vertex
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time)
{
    checkCudaErrors(cudaBindTextureToArray(tex, array));

    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel<<< grid, block>>>(pos, mesh_width, mesh_height, time);

    checkCudaErrors(cudaUnbindTexture(tex));
}

extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp)
{
    cudaChannelFormatDesc desc;

    if (Bpp == 1)
    {
        desc = cudaCreateChannelDesc<unsigned char>();
    }
    else
    {
        desc = cudaCreateChannelDesc<uchar4>();
    }

    checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
    checkCudaErrors(cudaMemcpyToArray(array, 0, 0, data, Bpp*sizeof(Pixel)*iw*ih, cudaMemcpyHostToDevice));
}

extern "C" void deleteTexture(void)
{
    checkCudaErrors(cudaFreeArray(array));
}


// Wrapper for the __global__ call that sets up the texture and threads
extern "C" void sobelFilter(Pixel *odata, int iw, int ih, float fScale)
{
    checkCudaErrors(cudaBindTextureToArray(tex, array));

    // Run the Sobel kernel
    //SobelTex<<<ih, 384>>>(odata, iw, iw, ih, fScale);

    checkCudaErrors(cudaUnbindTexture(tex));
}

#endif // #ifndef _SIMPLEGL_KERNEL_H_
