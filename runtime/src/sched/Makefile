.PHONY: all

TARGET := bemps_sched

SRC    := $(shell find -type f -name \*.cpp -print)
OBJ    := $(patsubst %.cpp,$(SCHED_OBJDIR)/%.o,$(SRC))



all: $(SCHED_OBJDIR)/$(TARGET)


$(SCHED_OBJDIR)/$(TARGET): $(OBJ)
	$(CXX) $(CFLAGS) -L$(BEMPS_OBJDIR) $^ -o $@ -lbemps -lrt -pthread

$(SCHED_OBJDIR)/%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ -D$(GPU_RES)

