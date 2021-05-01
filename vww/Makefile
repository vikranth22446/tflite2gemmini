include $(abs_top_srcdir)/Makefrag

tests = \
	main

tests_baremetal = $(tests:=-baremetal)
runs_baremetal = $(addsuffix .run,$(tests_baremetal))
tests_linux-pk = $(tests:=-linux-pk)

BENCH_COMMON = $(abs_top_srcdir)/riscv-tests/benchmarks/common
GEMMINI_HEADERS = $(abs_top_srcdir)/include/gemmini.h $(abs_top_srcdir)/include/gemmini_params.h $(abs_top_srcdir)/include/gemmini_nn.h $(abs_top_srcdir)/include/gemmini_testutils.h

CFLAGS := $(CFLAGS) \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=c++11 \
	-lgcc \
	-fpermissive \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-march=rv64gc -Wa,-march=rv64gcxhwacha \
	-I$(abs_top_srcdir)/riscv-tests \
	-I$(abs_top_srcdir)/riscv-tests/env \
	-I$(abs_top_srcdir) \
	-I$(BENCH_COMMON) \
	-I$(src_dir)/../../ \
	-I$(src_dir) \
	-DID_STRING=$(ID_STRING)

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-static-libgcc \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1 \

all: $(tests_baremetal) $(tests_linux-pk)

vpath %.cpp $(src_dir)/../../ $(src_dir)/../../api/

%-baremetal:  $(src_dir)/submitter_implemented.c $(GEMMINI_HEADERS)
	$(CC_BAREMETAL) $(CFLAGS_BAREMETAL) $< $(LFLAGS) -o $@ \
		$(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S) $(LIBS) $(src_dir)/../../main.cpp $(src_dir)/../../api/internally_implemented.cpp

%-linux-pk:  $(src_dir)/submitter_implemented.c $(GEMMINI_HEADERS)
	$(CC_LINUX) $(CFLAGS) -static $< $(LFLAGS) -o $@ \
		$(src_dir)/../../main.cpp $(src_dir)/../../api/internally_implemented.cpp

run-baremetal: $(runs_baremetal)

%-baremetal.run: %-baremetal
	$(RUNNER)$^

junk += $(tests_baremetal) $(tests_linux)

