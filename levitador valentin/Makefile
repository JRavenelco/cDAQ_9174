CXX = g++
CXXFLAGS = -O2 -Wall
SRCDIR = .
TARGETS = levitador_obs_minimal levitador_sensorless_final levitador_sensorless_kan levitador_serial_linux levitador_uol_ad1_monitor levitador_kan_simple

all: $(TARGETS)

levitador_kan_simple: $(SRCDIR)/levitador_kan_simple.cpp sensorless_hippo_kan_final.h CBR_InitPosition.h
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

levitador_obs_minimal: $(SRCDIR)/levitador_obs_minimal.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

levitador_sensorless_final: $(SRCDIR)/levitador_sensorless_final.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

levitador_sensorless_kan: $(SRCDIR)/levitador_sensorless_kan.cpp sensorless_hippo_kan_final.h CBR_InitPosition.h
	$(CXX) $(CXXFLAGS) -o $@ $<

levitador_serial_linux: $(SRCDIR)/levitador_serial_linux.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

levitador_uol_ad1_monitor: $(SRCDIR)/levitador_uol_ad1_monitor.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm -ldwf

clean:
	rm -f $(TARGETS)

.PHONY: all clean
