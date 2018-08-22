# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pavlin/dev/FIt-SNE

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pavlin/dev/FIt-SNE

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/cmake-gui -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/pavlin/dev/FIt-SNE/CMakeFiles /home/pavlin/dev/FIt-SNE/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/pavlin/dev/FIt-SNE/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named bin/fast_tsne

# Build rule for target.
bin/fast_tsne: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 bin/fast_tsne
.PHONY : bin/fast_tsne

# fast build rule for target.
bin/fast_tsne/fast:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/build
.PHONY : bin/fast_tsne/fast

# target to build an object file
src/nbodyfft.o:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/nbodyfft.o
.PHONY : src/nbodyfft.o

# target to preprocess a source file
src/nbodyfft.i:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/nbodyfft.i
.PHONY : src/nbodyfft.i

# target to generate assembly for a file
src/nbodyfft.s:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/nbodyfft.s
.PHONY : src/nbodyfft.s

# target to build an object file
src/sptree.o:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/sptree.o
.PHONY : src/sptree.o

# target to preprocess a source file
src/sptree.i:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/sptree.i
.PHONY : src/sptree.i

# target to generate assembly for a file
src/sptree.s:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/sptree.s
.PHONY : src/sptree.s

# target to build an object file
src/tsne.o:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/tsne.o
.PHONY : src/tsne.o

# target to preprocess a source file
src/tsne.i:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/tsne.i
.PHONY : src/tsne.i

# target to generate assembly for a file
src/tsne.s:
	$(MAKE) -f CMakeFiles/bin/fast_tsne.dir/build.make CMakeFiles/bin/fast_tsne.dir/src/tsne.s
.PHONY : src/tsne.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... bin/fast_tsne"
	@echo "... edit_cache"
	@echo "... src/nbodyfft.o"
	@echo "... src/nbodyfft.i"
	@echo "... src/nbodyfft.s"
	@echo "... src/sptree.o"
	@echo "... src/sptree.i"
	@echo "... src/sptree.s"
	@echo "... src/tsne.o"
	@echo "... src/tsne.i"
	@echo "... src/tsne.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
