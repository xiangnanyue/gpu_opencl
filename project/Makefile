EXE=videofilter_gpu
CVINCFLAGS=`pkg-config --libs --cflags opencv`
CVLIBFLAGS=`pkg-config --libs opencv`
DBGFLAGS= -pg
MGD=/opt/Mali_Graphics_Debugger_v4.4.1.0271762a_Linux_x64/target/linux/hard_float/
GCC=g++
SRCS=./videofilter_gpu.cpp
OCLLIBSDIR=/opt/ComputeLibrary/build/
OCLINCSDIR=/opt/ComputeLibrary/include/ 
FLAGS=-g -Wno-deprecated-declarations -Wall -DARCH_ARM -Wextra -Wno-unused-parameter -pedantic -Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default -fpermissive -std=gnu++11 -Wno-vla -Woverloaded-virtual -Wctor-dtor-privacy -Wsign-promo -Weffc++ -Wno-format-nonliteral -Wno-overlength-strings -Wno-strict-overflow -Wno-implicit-fallthrough -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -march=armv7-a -mthumb -mfpu=neon -mfloat-abi=hard -Werror -O3 -ftree-vectorize -fstack-protector-strong -DARM_COMPUTE_CL -I${OCLINCSDIR} -I.. -I..  
LDFLAGS=-L${OCLLIBSDIR} -larm_compute -larm_compute_core -lOpenCL
all:${EXE}
${EXE}: ${SRCS}
	${GCC}  ${FLAGS} ${DBGFLAGS} ${CVINCFLAGS} ${SRCS} ${CVLIBFLAGS}  -DARM_COMPUTE_CL ${LDFLAGS}   -o ${EXE}


profile:
	./${EXE}
	gprof ./${EXE} gmon.out > prof.txt

debug:${EXE}
	LD_PRELOAD=${MGD}/libinterceptor.so ./${EXE}

clean:
	rm -rf ${EXE} *.o
