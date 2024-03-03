cmake_minimum_required(VERSION 3.27)
project(IA C)

set(CMAKE_C_STANDARD 11)

add_executable(IA main.c
        functions.c
        functions.h
        Lecture/Bmp2Matrix.c
        Lecture/Bmp2Matrix.h
)

