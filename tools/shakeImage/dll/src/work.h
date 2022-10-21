#pragma once

#ifndef _LIB_EXPORT
#define _LIB_EXPORT

#ifdef _MSC_VER
#    define LIBEXPORT __declspec(dllexport)
#    define LIBIMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#    define LIBEXPORT __attribute__((visibility("default")))
#    define LIBIMPORT
#else
#    define LIBEXPORT
#    define EXPIMP_TEMPLATE
#pragma warning Unknown dynamic link import/export semantics.
#endif 
#ifdef NDEBUG
#define ASSERT(x) { if(!(x)) throw "Exception!!!";}
#else
#include <assert.h>
#define ASSERT(x) assert(x)
#endif

#endif// _LIB_EXPORT

#ifdef __cplusplus
extern "C"
{
#endif
	typedef unsigned char uchar;
	LIBEXPORT  unsigned char*LoadFrame(uchar* fisrtFrame, uchar* currentFrame, int height, int width, int channels);
	LIBEXPORT  void ReleaseFrame(unsigned char* data);
	LIBEXPORT void showNdarray(int* data, int rows, int cols);
	LIBEXPORT bool TestVideoShake(char* intputfilepath, char* outputfilepath);

#ifdef __cplusplus
}
#endif