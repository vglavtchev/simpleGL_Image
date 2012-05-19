////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <rendercheck_gl.h>
#include <helper_functions.h>
//#include <helper_cuda.h>      // includes for cuda initialization and error checking
#include <shrQATest.h>    // standard utility and system includes
#include <vector_types.h>
#include <limits.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD         0.30f
#define REFRESH_DELAY     10 //ms

typedef unsigned char Pixel;

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

unsigned int mesh_width = 1024;
unsigned int mesh_height = 1024;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_Verify = false;
bool g_bQAReadback = false;
bool g_bGLVerify   = false;

int *pArgc = NULL;
char **pArgv = NULL;


/*
 * ------ GLOBALS FROM SobelFilter
 */

unsigned int g_Bpp;

// Display Data
static GLuint pbo_buffer = 0;  // Front and back CA buffers
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

static GLuint texid = 0;       // Texture for display
float imageScale = 1.f;        // Image exposure

// Create containers for the original image, the histogram, and the HistEq image
unsigned int gHistogram[UCHAR_MAX-1];
unsigned char *pixels = NULL;  // Image pixel data on the host
unsigned char *hEqImage = NULL;

/*
 *
 *
 */

// CheckFBO/BackBuffer class objects
CheckRender       *g_CheckRender = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// kernels
//#include <simpleGL_kernel.cu>

extern "C" void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);

extern "C" void sobelFilter(Pixel *odata, int iw, int ih, float fScale);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);
void loadDefaultImage(char *loc_exec);
void initializeData(char *file);
void histogramCreate_CPU(const unsigned char *pixels);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest();
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *SDK_name = "simpleGL (VBO)";
char* g_cCurrentDeviceName;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

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

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }

    if (devID < 0)
    {
        devID = 0;
    }

    if (devID > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        exit(-1);
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);

    return devID;
}

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    printf("MapSMtoCores undefined SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
    return -1;
}
// end of GPU Architecture definitions

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_compute_perf   = 0, max_perf_device   = 0;
    int device_count       = 0, best_SM_arch      = 0;
    int best_SM_minor = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        printf("Device name: %s - DeviceProp.major = %d\n", deviceProp.name, deviceProp.major);
        if (deviceProp.major > 0 && deviceProp.major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, deviceProp.major);
            best_SM_minor = MAX(best_SM_minor, deviceProp.minor);
        }

        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        printf("Device found: %s\n", deviceProp.name);

        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        }

        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

        printf("Current device compute performance: %d\n", compute_perf);
        if (compute_perf  > max_compute_perf)
        {
			max_compute_perf  = compute_perf;
			max_perf_device   = current_device;
			printf("Current device has best perf\n");
        }

        ++current_device;
    }

    // Get the name of the best performing device
    cudaGetDeviceProperties(&deviceProp, max_perf_device);
    g_cCurrentDeviceName = (char*)malloc(strlen(deviceProp.name));
    strcpy(g_cCurrentDeviceName, deviceProp.name);
    printf("Max performance device: %d - %s\n", max_perf_device, g_cCurrentDeviceName);

    return max_perf_device;
}


// Initialization code to find the best CUDA Device
int findCudaDevice(int argc, const char **argv)
{
    cudaDeviceProp deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");

        if (devID < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(-1);
        }
        else
        {
            devID = gpuDeviceInit(devID);

            if (devID < 0)
            {
                printf("exiting...\n");
                shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                exit(-1);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors(cudaSetDevice(devID));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    return devID;
}

inline int gpuGLDeviceInit(int ARGC, char **ARGV)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }

    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuGLDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "Error: device does not support CUDA.\n");
        exit(-1);
        \
    }

    if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == false)
    {
        fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);
    }

    checkCudaErrors(cudaGLSetGLDevice(dev));
    return dev;
}

// This function will pick the best CUDA device available with OpenGL interop
inline int findCudaGLDevice(int argc, char **argv)
{
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = gpuGLDeviceInit(argc, argv);

        if (devID < 0)
        {
            printf("exiting...\n");
            cudaDeviceReset();
            exit(0);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        cudaGLSetGLDevice(devID);
    }

    return devID;
}

////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return true if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
inline bool
sdkCheckErrorGL(const char *file, const int line)
{
    bool ret_val = true;

    // check for error
    GLenum gl_error = glGetError();

    if (gl_error != GL_NO_ERROR)
    {
#ifdef _WIN32
        char tmpStr[512];
        // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
        // when the user double clicks on the error line in the Output pane. Like any compile error.
        sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, gluErrorString(gl_error));
        OutputDebugString(tmpStr);
#endif
        fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
        fprintf(stderr, "%s\n", gluErrorString(gl_error));
        ret_val = false;
    }

    return ret_val;
}

#define SDK_CHECK_ERROR_GL()                                              \
    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
        exit(EXIT_FAILURE);                                               \
    }
// end of CUDA Helper Functions

bool checkHW(char *name, char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        shrQAFinishExit(*pArgc, (const char **)pArgv, QA_FAILED);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, "Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

void equalizeImage(const unsigned char* pixels){

	// Create output image
	hEqImage = (unsigned char*)malloc( sizeof(unsigned char) * mesh_width * mesh_height);

	unsigned int cdf[UCHAR_MAX];
	unsigned int runningSum = 0;

	for(int i = 0; i < UCHAR_MAX-1; i++){
		runningSum += gHistogram[i];
		cdf[i] = runningSum;
	}

	// Calculate equalized image
	unsigned int totalElements = (unsigned int)mesh_width * (unsigned int)mesh_height;

	unsigned int currentValue = 0;
	unsigned int currentResult = 0;

	for(int i = 0; i < mesh_width * mesh_height; i++){
		currentValue = cdf[ pixels[i] ];
		currentResult = (currentValue * (UCHAR_MAX-1)) / totalElements;
		hEqImage[i] = (unsigned char)currentResult;
	}
	printf("Finished equalizing\n");

    //FILE *fp;
    //fopen(fp, "lena_eq.pgm", "wb");
	char *saveFile = "./data/lena_eq.pgm";
	sdkSavePGM(saveFile, hEqImage, mesh_width, mesh_height);
}

void histogramCreate_CPU(const unsigned char *pixels)
{
	// Clear out the histogram counts
    memset(gHistogram, 0x0, sizeof(Pixel) * UCHAR_MAX);

    // Create histogram
	for(int i = 0; i < mesh_width * mesh_height; i++){
		gHistogram[ pixels[i] ]++;
	}

	// Print the histogram in the console
	/*int count = 0;
	for(int i = 0; i < UCHAR_MAX; i++){
		count = gHistogram[i];

		// print a '|' character for each 10 counts for a value
		if(count > 10){
			printf("[%d]: |", i);
			for(int i = 0; i < count; i++){
				if(count % 10 == 0)
					printf("|");
			}
			printf("\n");
		}
	}*/
}

void initializeData(char *file)
{
    GLint bsize;
    unsigned int w, h;
    size_t file_length= strlen(file);

    // Load one frame of image
    if (!strcmp(&file[file_length-3], "pgm"))
    {
    	printf("Filename: %s \n", file);
        if (sdkLoadPGM<unsigned char>(file, &pixels, &w, &h) != true)
        {
            printf("Failed to load PGM image file: %s\n", file);
            exit(-1);
        }
        printf("Loaded image successfully.\n");

        g_Bpp = 1;
    }

    // Create a texture and load the image onto it
    mesh_width = (int)w;
    mesh_height = (int)h;
    setupTexture(mesh_width, mesh_height, pixels, g_Bpp);
    printf("Texture setup completed. Size: %d x %d\n", mesh_width, mesh_height);

    // Create histogram
    histogramCreate_CPU(pixels);
    equalizeImage(pixels);

    //memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * mesh_width * mesh_height);

    if (!g_bQAReadback)
    {
        printf("Starting PBO setup.\n");

        // This code creates a PBO for the output image
        // But in our application it is NOT necessary
        /*glGenBuffers(1, &pbo_buffer);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
        glBufferData(GL_PIXEL_UNPACK_BUFFER,
                     g_Bpp * sizeof(Pixel) * mesh_width * mesh_height,
                     pixels, GL_STREAM_DRAW);

        glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);

        if ((GLuint)bsize != (g_Bpp * sizeof(Pixel) * mesh_width * mesh_height))
        {
            printf("Buffer object (%d) has incorrect size (%d).\n", (unsigned)pbo_buffer, (unsigned)bsize);
            cudaDeviceReset();
            exit(-1);
        }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // register this buffer object with CUDA
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));
		*/

        glGenTextures(1, &texid);
        glBindTexture(GL_TEXTURE_2D, texid);
        glTexImage2D(GL_TEXTURE_2D, 0, ((g_Bpp==1) ? GL_LUMINANCE : GL_BGRA),
        		mesh_width, mesh_height,  0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

    }
}

void loadDefaultImage(char *loc_exec)
{

    printf("Reading image: lena.pgm\n");
    const char *image_filename = "lena.pgm";
    char *image_path = sdkFindFilePath(image_filename, loc_exec);

    if (image_path == NULL)
    {
        printf("Failed to read image file: <%s>\n", image_filename);
        shrQAFinishExit2(false, *pArgc, (const char **)pArgv, QA_FAILED);
    }

    initializeData(image_path);
    free(image_path);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    shrQAStart(argc, argv);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "qatest") ||
            checkCmdLineFlag(argc, (const char **)argv, "noprompt"))
        {
            printf("- (automated test no-OpenGL)\n");
            g_bQAReadback = true;
            //          g_bGLVerify = true;
            fpsLimit = frameCheckNumber;
        }
        else if (checkCmdLineFlag(argc, (const char **)argv, "glverify"))
        {
            printf("- (automated test OpenGL rendering)\n");
            g_bGLVerify = true;
            fpsLimit = frameCheckNumber;
        }
    }

    printf("\n");

    runTest(argc, argv);

    cudaDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (g_TotalErrors == 0) ? QA_PASSED : QA_FAILED);
}

void AutoQATest()
{
    if (g_CheckRender && g_CheckRender->IsQAReadback())
    {
        char temp[256];
        sprintf(temp, "AutoTest: Cuda GL Interop (VBO)");
        glutSetWindowTitle(temp);
        shrQAFinishExit2(true, *pArgc, (const char **)pArgv, QA_PASSED);
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit-1)
    {
        g_Verify = true;
    }

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);

        if(frameCount % 10 == 0){
			sprintf(fps, "%sSimpleGL_Image: %3.1f fps on device [%s]",
					((g_CheckRender && g_CheckRender->IsQAReadback()) ? "AutoTest: " : ""), ifps, g_cCurrentDeviceName);
			glutSetWindowTitle(fps);
        }
        fpsCount = 0;

        if (g_CheckRender && !g_CheckRender->IsQAReadback())
        {
            fpsLimit = (int)MAX(ifps, 1.f);
        }

        sdkResetTimer(&timer);

        AutoQATest();
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    printf("Finished loading image\n");

    // command line mode only
    if (g_bQAReadback)
    {
        // This will pick the best possible CUDA capable device
        int devID = findCudaDevice((const int)argc, (const char **)argv);

        // create VBO
        createVBO(NULL, NULL, 0);
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            gpuGLDeviceInit(argc, argv);
        }
        else
        {
            cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
        }

        // Load input image into a texture
        loadDefaultImage(argv[0]);

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
    }

    if (g_bQAReadback)
    {
        g_CheckRender = new CheckBackBuffer(window_width, window_height, 4, false);
        g_CheckRender->setPixelFormat(GL_RGBA);
        g_CheckRender->setExecPath(argv[0]);
        g_CheckRender->EnableQAReadback(true);

        runAutoTest();
    }
    else
    {
        if (g_bGLVerify)
        {
            g_CheckRender = new CheckBackBuffer(window_width, window_height, 4);
            g_CheckRender->setPixelFormat(GL_RGBA);
            g_CheckRender->setExecPath(argv[0]);
            g_CheckRender->EnableQAReadback(true);
        }

        // run the cuda part
        runCuda(&cuda_vbo_resource);
    }

    // check result of Cuda step
    checkResultCuda(argc, argv, vbo);

    if (!g_bQAReadback)
    {
        atexit(cleanup);

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    // DEPRECATED: checkCudaErrors(cudaGLMapBufferObject((void**)&dptr, vbo));
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    // execute the kernel
    //    dim3 block(8, 8, 1);
    //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(vbo));
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest()
{
    // execute the kernel
    launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(g_CheckRender->imageData(), d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));
    g_CheckRender->dumpBin((void *)g_CheckRender->imageData(), mesh_width*mesh_height*sizeof(float), "simpleGL.bin");

    if (!g_CheckRender->compareBin2BinFloat("simpleGL.bin", "ref_simpleGL.bin", mesh_width*mesh_height*sizeof(float), MAX_EPSILON_ERROR, THRESHOLD))
    {
        g_TotalErrors++;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    if (vbo)
    {
        // create buffer object
        glGenBuffers(1, vbo);
        glBindBuffer(GL_ARRAY_BUFFER, *vbo);

        // initialize buffer object
        unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // register this buffer object with CUDA
        // DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(*vbo));
        printf("Registering VBO with CUDA\n");
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

        SDK_CHECK_ERROR_GL();
    }
    else
    {
    	printf("Allocating VBO\n");
        checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float)));
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
    if (vbo)
    {
        // unregister this buffer object with CUDA
        //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(*pbo));
        cudaGraphicsUnregisterResource(vbo_res);

        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);

        *vbo = 0;
    }
    else
    {
        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(0.0, 0.5, 0.5);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &pbo_buffer);
    glDeleteTextures(1, &texid);
    deleteTexture();
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    deleteVBO(&vbo, cuda_vbo_resource);

    if (g_CheckRender)
    {
        delete g_CheckRender;
        g_CheckRender = NULL;
    }
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            exit(0);
            break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(vbo));
        cudaGraphicsUnregisterResource(cuda_vbo_resource);

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER_ARB, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (! glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        //DEPRECATED: checkCudaErrors(cudaGLRegisterBufferObject(vbo));
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}
