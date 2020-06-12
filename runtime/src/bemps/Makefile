.PHONY: all

override CFLAGS += -fPIC

TARGET := libbemps.so

SRC    :=   $(shell find -type f -name \*.cpp -print)
OBJ    :=   $(patsubst %.cpp,$(BEMPS_OBJDIR)/%.o,$(SRC))



all: $(BEMPS_OBJDIR)/$(TARGET)


$(BEMPS_OBJDIR)/$(TARGET): $(OBJ)
	$(CXX) $(CFLAGS) -shared $^ -o $@ -L$(CUDA_LIB) -lrt -lcudart

$(BEMPS_OBJDIR)/%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(CUDA_INC) -c $< -o $@

