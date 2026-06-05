cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(Tumor_Segmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/Derng-nn/Tumor_Segmentation.git")
set(EXTENSION_CONTRIBUTORS "Noppanon Nobnop (Image Lab.)")
set(EXTENSION_DESCRIPTION "Pelvic tumor segmentation")
set(EXTENSION_ICONURL "https://github.com/Derng-nn/Tumor_Segmentation/Tumor_Segmentation/Resources/Icons/Tumor_Segmentation.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA") 

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Tumor_Segmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
