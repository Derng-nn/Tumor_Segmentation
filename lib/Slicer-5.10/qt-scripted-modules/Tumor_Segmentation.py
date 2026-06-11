import logging
import os
import slicer

# -----------------------------
# Auto install required packages
# -----------------------------
def ensurePythonPackage(packageName, importName=None):
    if importName is None:
        importName = packageName

    try:
        __import__(importName)
    except ImportError:
        slicer.util.infoDisplay(
            f"Installing required package: {packageName}\n"
            "This may take a few minutes."
        )

        slicer.util.pip_install(packageName)

        try:
            __import__(importName)
        except ImportError:
            raise RuntimeError(
                f"Failed to install Python package: {packageName}"
            )

# Install if missing
ensurePythonPackage("onnxruntime")
ensurePythonPackage("opencv-python", "cv2")

# Normal imports
import numpy as np
import onnxruntime as ort
import cv2

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode


#
# Tumor_Segmentation
#


class Tumor_Segmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Tumor_Segmentation")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Noppanon Nobnop (Department of Biomedical Engineering Srinakharinwirot University.)"]  # TODOS: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
3D Slicer extension for tumor segmentation in MRI using "Tumor_Segmentator" AI model.                                

See more information in <a href="https://github.com/organization/projectname#Tumor_Segmentation">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Noppanon Nobnop (ImageLab, Srinakharinwirot University).
If you use the Tumor_Segmentator from this software in your research, please cite:                                      
N. Nobnop, N. Yamcharoen, C. Sukjamsri, T. Piboonthummasak, T. Charoenpong and P. Kiatisevi, 
"Pelvic Tumor Segmentation in MRI Images Using Deep Learning with DeepLabV3+ and U-Net: A Performance Comparison," 
doi: 10.1109/BMEiCON64021.2024.10896343. and N. Nobnop, P. Kiatisevi, C. Sukjamsri and T. Charoenpong, 
"Pelvic Tumor Segmentation in Magnetic Resonance Images By U-Net," doi: 10.1109/ICIIBMS66230.2025.11316723.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Tumor_Segmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Tumor_Segmentation",
        sampleName="Tumor_Segmentation1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "Tumor_Segmentation1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="Tumor_Segmentation1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="Tumor_Segmentation1",
    )

    # Tumor_Segmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Tumor_Segmentation",
        sampleName="Tumor_Segmentation2",
        thumbnailFileName=os.path.join(iconsPath, "Tumor_Segmentation2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="Tumor_Segmentation2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="Tumor_Segmentation2",
    )


#
# Tumor_SegmentationParameterNode
#


@parameterNodeWrapper
class Tumor_SegmentationParameterNode:

    inputVolume: vtkMRMLScalarVolumeNode
    outputVolume: vtkMRMLScalarVolumeNode
    startSlice: int
    endSlice: int

#
# Tumor_SegmentationWidget
#


class Tumor_SegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

        self.startSlice = None
        self.endSlice = None
        self.roiNode = None


    def setup(self):

        ScriptedLoadableModuleWidget.setup(self)

        #Path setup
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Tumor_Segmentation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = Tumor_SegmentationLogic() #call logic

        # -----------------------------
        # Segment Editor setup
        # -----------------------------
        self.ui.roiSegmentEditor.setMRMLScene(slicer.mrmlScene)
        self.roiEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.ui.roiSegmentEditor.setMRMLSegmentEditorNode(self.roiEditorNode)

        # -----------------------------
        # Connections
        # -----------------------------
        self.ui.inputSelector.currentNodeChanged.connect(self.onInputChanged)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.show3DButton.connect("clicked(bool)", self.onShow3DButton)
        self.ui.createRoiButton.clicked.connect(self.onCreateRoi)
        self.ui.useRoiCheckBox.connect("toggled(bool)",self.onUseRoiToggled)
        self.ui.setStartSliceButton.connect("clicked(bool)",self.onSetStartSlice)
        self.ui.setEndSliceButton.connect("clicked(bool)",self.onSetEndSlice)

        self.ui.show3DButton.enabled = False
        self.ui.setStartSliceButton.enabled = True
        self.ui.setEndSliceButton.enabled = True
        self.ui.createRoiButton.enabled = False
        self.initializeParameterNode()


    def onInputChanged(self, node):

        if node and self.roiNode:
            self.ui.roiSegmentEditor.setSourceVolumeNode(node)


    def onUseRoiToggled(self, checked):

        self.ui.createRoiButton.enabled = checked
        #self.ui.setStartSliceButton.enabled = checked
        #self.ui.setEndSliceButton.enabled = checked
        #self.ui.createRoiButton.enabled = checked


    def onSetStartSlice(self):

        lm = slicer.app.layoutManager()
        redWidget = lm.sliceWidget("Red")
        sliceLogic = redWidget.sliceLogic()

        sliceIndex = sliceLogic.GetSliceIndexFromOffset(
            sliceLogic.GetSliceOffset()
        )

        self.startSlice = sliceIndex
        self.ui.startSliceLabel.setText(f"Start: {sliceIndex}")


    def onSetEndSlice(self):

        lm = slicer.app.layoutManager()
        redWidget = lm.sliceWidget("Red")
        sliceLogic = redWidget.sliceLogic()

        sliceIndex = sliceLogic.GetSliceIndexFromOffset(
            sliceLogic.GetSliceOffset()
        )

        self.endSlice = sliceIndex
        self.ui.endSliceLabel.setText(f"End: {sliceIndex}")


    def cleanup(self):
        self.removeObservers()


    def enter(self):
        self.initializeParameterNode()


    def exit(self):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._checkCanApply
            )


    def initializeParameterNode(self):

        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.inputVolume:

            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass(
                "vtkMRMLScalarVolumeNode"
            )

            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode


    def setParameterNode(self, inputParameterNode):

        if self._parameterNode:

            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)

            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._checkCanApply
            )

        self._parameterNode = inputParameterNode

        if self._parameterNode:

            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self._checkCanApply
            )

            self._checkCanApply()


    def _checkCanApply(self, caller=None, event=None):

        if (
            self._parameterNode
            and self._parameterNode.inputVolume
            and self._parameterNode.outputVolume
        ):

            self.ui.applyButton.enabled = True

        else:

            self.ui.applyButton.enabled = False


    def onApplyButton(self):

        with slicer.util.tryWithErrorDisplay(
            _("Tumor segmentation failed."),
            waitCursor=True
        ):

            inputNode = self.ui.inputSelector.currentNode()

            if not inputNode:
                raise RuntimeError("Please select input volume")

            vol = slicer.util.arrayFromVolume(inputNode)
            depth = vol.shape[0]

            # -----------------------------
            # Slice range
            # -----------------------------
            if self.startSlice is None:
                startSlice = 0
            else:
                startSlice = self.startSlice

            if self.endSlice is None:
                endSlice = depth - 1
            else:
                endSlice = self.endSlice

            if startSlice > endSlice:
                raise RuntimeError("Start slice must be smaller than End slice")

            # -----------------------------
            # ROI usage
            # -----------------------------
            roiNode = None

            if self.ui.useRoiCheckBox.checked:

                if self.roiNode is None:
                    raise RuntimeError("Please create ROI first")

                roiNode = self.roiNode

            # -----------------------------
            # Run segmentation
            # -----------------------------
            segNode = self.logic.process(
                inputVolume=inputNode,
                outputVolume=None,
                startSlice=startSlice,
                endSlice=endSlice,
                roiNode=roiNode
            )

            # -----------------------------
            # Display result
            # -----------------------------
            self.logic._showAxialOnly(inputNode, segNode)

            self._currentSegNode = segNode
            self.ui.show3DButton.enabled = True


    def onCreateRoi(self):

        inputNode = self.ui.inputSelector.currentNode()

        if not inputNode:
            slicer.util.errorDisplay("Please select input volume first")
            return

        if self.startSlice is None or self.endSlice is None:
            slicer.util.errorDisplay("Please set Start / End slice first")
            return

        self.roiNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            "ROI_Segmentation"
        )

        self.roiNode.CreateDefaultDisplayNodes()

        self.roiNode.SetReferenceImageGeometryParameterFromVolumeNode(
            inputNode
        )

        self.roiNode.GetSegmentation().AddEmptySegment("ROI")

        self.ui.roiSegmentEditor.setSegmentationNode(self.roiNode)
        self.ui.roiSegmentEditor.setSourceVolumeNode(inputNode)

        slicer.util.infoDisplay("Draw ROI using Paint or Draw tool")


    def onShow3DButton(self):

        segNode = getattr(self, "_currentSegNode", None)

        if not segNode:
            return

        displayNode = segNode.GetDisplayNode()

        if not displayNode:
            return

        displayNode.RemoveAllViewNodeIDs()

        displayNode.SetVisibility(True)
        displayNode.SetVisibility3D(True)

        slicer.app.layoutManager().setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView
        )

        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()

#
# Tumor_SegmentationLogic
#


class Tumor_SegmentationLogic(ScriptedLoadableModuleLogic):

    def __init__(self):

        ScriptedLoadableModuleLogic.__init__(self)

        import onnxruntime as ort
        import os

        self.session = None
        self.inputName = None
        self.modelH = 256
        self.modelW = 256

        try:

            modelPath = os.path.join(
                os.path.dirname(__file__),
                "Resources",
                "Models",
                "unet.onnx"
            )
            print("Loading ONNX model...")
            print("Model path:", modelPath)
            print("File exists:", os.path.exists(modelPath))

            self.session = ort.InferenceSession(
                modelPath,
                providers=["CPUExecutionProvider"]
            )

            self.inputName = self.session.get_inputs()[0].name

            print("ONNX model loaded successfully")

        except Exception as e:

            print("Failed to load ONNX model")
            print(e)


    def _ensureSession(self):
        if self.session is not None:
            return

        import os
        import onnxruntime as ort

        self.modelPath = os.path.join(
            os.path.dirname(__file__),
            "Resources", "Models", "unet.onnx"
        )

        self.session = ort.InferenceSession(
            self.modelPath,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        inp = self.session.get_inputs()[0]
        self.inputName = inp.name
        shape = inp.shape
        self.modelH = shape[2] or 256
        self.modelW = shape[3] or 256


    def _showAxialOnly(self, volumeNode, segNode):

        lm = slicer.app.layoutManager()
        redWidget = lm.sliceWidget('Red')
        greenWidget = lm.sliceWidget('Green')
        yellowWidget = lm.sliceWidget('Yellow')

        redSliceNode = redWidget.mrmlSliceNode()
        greenSliceNode = greenWidget.mrmlSliceNode()
        yellowSliceNode = yellowWidget.mrmlSliceNode()

        # Background volume
        redWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(
            volumeNode.GetID()
        )
        greenWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(None)
        yellowWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(None)

        # Segmentation display node
        displayNode = segNode.GetDisplayNode()
        if not displayNode:
            displayNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationDisplayNode"
            )
            segNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        # Basic visibility
        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)
        displayNode.SetVisibility3D(False)
        displayNode.SetVisibility2DFill(True)
        displayNode.SetVisibility2DOutline(True)

        # show ONLY in Red slice
        displayNode.AddViewNodeID(redSliceNode.GetID())

        # Force axial
        redSliceNode.SetOrientationToAxial()
        redWidget.sliceLogic().FitSliceToAll()

    def rasToIjk(self, volumeNode, sliceOffset):

        ras = [0, 0, sliceOffset, 1]

        ijk = [0, 0, 0, 0]

        volumeNode.GetRASToIJKMatrix().MultiplyPoint(ras, ijk)

        return ijk

    # -------------------------
    # Axial reslice
    # -------------------------
    def _resliceToAxial(self, inputVolume):
        """
        Force volume to Axial orientation using ResampleScalarVolume (correct CLI usage)
        """

        axialVolume = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode",
            "AxialReslicedTemp"
        )

        params = {
            "inputVolume": inputVolume.GetID(),
            "outputVolume": axialVolume.GetID(),
            "orientation": "Axial",
            "interpolationType": "linear"
        }

        slicer.cli.runSync(
            slicer.modules.resamplescalarvolume,
            None,
            params
        )

        # IMPORTANT: force MRML update
        axialVolume.GetImageData().Modified()
        axialVolume.Modified()

        if axialVolume.GetImageData() is None:
            raise RuntimeError("Axial reslicing failed: output volume has no image data")

        return axialVolume

    def getParameterNode(self):
        return Tumor_SegmentationParameterNode(super().getParameterNode())

    # -------------------------
    # Main process
    # -------------------------
    def process(
            self,
            inputVolume,
            outputVolume=None,
            startSlice=0,
            endSlice=None,
            roiNode=None
        ):

        import numpy as np
        import cv2
        import slicer
        import vtk

        logging.info("Starting tumor segmentation")

        # -----------------------------
        # Ensure ONNX session
        # -----------------------------
        self._ensureSession()

        # -----------------------------
        # Load volume
        # -----------------------------
        vol = slicer.util.arrayFromVolume(inputVolume)

        depth, height, width = vol.shape

        if endSlice is None:
            endSlice = depth - 1

        outputMask = np.zeros_like(vol, dtype=np.uint8)

        # -----------------------------
        # ROI mask
        # -----------------------------
        roiMask = None

        if roiNode is not None:

            roiMask = slicer.util.arrayFromSegmentBinaryLabelmap(
                roiNode,
                "ROI",
                inputVolume
            )

        # -----------------------------
        # Slice loop
        # -----------------------------
        for i in range(startSlice, endSlice + 1):

            sliceImg = vol[i]

            # -------------------------
            # Preprocess (same as test code)
            # -------------------------
            img = cv2.resize(
                sliceImg,
                (self.modelW, self.modelH)
            )

            img = img.astype(np.float32)

            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)

            # -------------------------
            # ONNX inference
            # -------------------------
            outputs = self.session.run(
                None,
                {self.inputName: img}
            )

            raw_result = np.squeeze(outputs[0])

            # -------------------------
            # Softmax (same as test script)
            # -------------------------
            exp_x = np.exp(raw_result - np.max(raw_result, axis=0))
            prob_map = exp_x / exp_x.sum(axis=0)

            ch0_tumor = prob_map[0, :, :]
            ch1_background = prob_map[1, :, :]

            pred = (ch0_tumor > ch1_background).astype(np.uint8)

            # -------------------------
            # Resize back to original
            # -------------------------
            pred = cv2.resize(
                pred,
                (width, height),
                interpolation=cv2.INTER_NEAREST
            )

            # -------------------------
            # Apply ROI mask
            # -------------------------
            if roiMask is not None:

                roiSlice = roiMask[i]

                # normalize ROI
                roiSlice = (roiSlice > 0).astype(np.uint8)

                # ensure same shape
                if roiSlice.shape != pred.shape:
                    roiSlice = cv2.resize(
                        roiSlice,
                        (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                pred = pred * roiSlice

            # -------------------------
            # Store result
            # -------------------------
            outputMask[i] = pred

            print("Slice", i, "tumor pixels:", np.sum(pred))

        print("Final mask sum:", np.sum(outputMask))

        # -----------------------------
        # Create output volume
        # -----------------------------
        if outputVolume is None:

            outputVolume = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLScalarVolumeNode",
                "TumorMaskVolume"
            )

            outputVolume.CreateDefaultDisplayNodes()

        slicer.util.updateVolumeFromArray(outputVolume, outputMask)

        # -----------------------------
        # Create labelmap
        # -----------------------------
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            "TumorLabelMap"
        )

        slicer.util.updateVolumeFromArray(labelmapNode, outputMask)

        labelmapNode.SetSpacing(inputVolume.GetSpacing())
        labelmapNode.SetOrigin(inputVolume.GetOrigin())

        ijkToRAS = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRAS)
        labelmapNode.SetIJKToRASMatrix(ijkToRAS)

        # -----------------------------
        # Create segmentation node
        # -----------------------------
        segNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode",
            "TumorSegmentation"
        )

        segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapNode,
            segNode
        )

        # -------- set tumor color --------
        seg = segNode.GetSegmentation()
        segmentID = seg.GetNthSegmentID(0)
        segment = seg.GetSegment(segmentID)

        segment.SetColor(1.0, 0.0, 0.0)   # red

        slicer.mrmlScene.RemoveNode(labelmapNode)

        return segNode