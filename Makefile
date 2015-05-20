CXX=g++

EIGEN_LOCATION=XXXX #change here
BOOST_LOCATION=XXXX #change here
BUILD_DIR=objs

CXXFLAGS =-Wall
#CXXFLAGS+=-g
#CXXFLAGS+=-pg
CXXFLAGS+=-O3
CXXFLAGS+=-std=c++0x
CXXFLAGS+=-lm
CXXFLAGS+=-fomit-frame-pointer
CXXFLAGS+=-fno-schedule-insns2
CXXFLAGS+=-fexceptions
CXXFLAGS+=-funroll-loops
CXXFLAGS+=-march=native
CXXFLAGS+=-mfpmath=sse
CXXFLAGS+=-msse4.2
CXXFLAGS+=-mmmx
CXXFLAGS+=-fopenmp
CXXFLAGS+=-m64
CXXFLAGS+=-DEIGEN_DONT_PARALLELIZE
CXXFLAGS+=-DEIGEN_NO_DEBUG
#CXXFLAGS+=-DNDEBUG
CXXFLAGS+=-DEIGEN_NO_STATIC_ASSERT
CXXFLAGS+=-I$(EIGEN_LOCATION)
CXXFLAGS+=-I$(BOOST_LOCATION)

CXXLIBS =-lz $(BOOST_IO)
#CXXLIBS+=-pg
CXXLIBS+=-fopenmp
CXXLIBS+=-L$(BOOST_LOCATION)/libs

SRCS=$(shell ls *.cpp)
OBJS=$(SRCS:.cpp=.o)

PROGRAM=pas-clblm

all : $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,$(PROGRAM))

$(BUILD_DIR)/%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

$(BUILD_DIR)/$(PROGRAM) : $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
	$(CXX) $(CXXFLAGS) $(CXXLIBS) -o $@ $^
	mv $(BUILD_DIR)/$(PROGRAM) ./
	rm -f ?*~

clean:
	rm -f $(BUILD_DIR)/* $(PROGRAM) ?*~
