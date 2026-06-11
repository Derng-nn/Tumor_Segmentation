import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Annotated

import numpy as np
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper

from slicer import vtkMRMLScalarVolumeNode

# ====================================================================
# ระบบรักษาสิทธิ์และการตั้งค่าไฟล์ Log อัตโนมัติ (ป้องกันปัญหาตอนย้ายเครื่อง)
# ====================================================================
def setup_module_logger():
    """
    ตั้งค่า Logger ประจำโมดูล ให้เขียนไฟล์ Log อยู่ในโครงสร้างโฟลเดอร์ของตัวเองเสมอ
    และควบคุมขนาดไฟล์ไม่ให้บวมขึ้นเรื่อยๆ
    """
    logger_name = "TumorSegmentationLogger"
    logger = logging.getLogger(logger_name)
    
    # ถ้าถูก Initialize ไปแล้วในเซสชันนี้ ไม่ต้องสร้าง Handler ซ้ำ (ป้องกัน Log เบิ้ลบรรทัด)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    try:
        # บังคับหา Path สัมบูรณ์ของโฟลเดอร์ปัจจุบัน (Tumor_Segmentation/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, "Resources", "Logs")
        
        # สร้างโฟลเดอร์ Logs มารองรับถ้ายังไม่มี
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        log_path = os.path.join(log_dir, "tumor_segmentation.log")
        
        # จำกัดขนาดไฟล์ที่ 500KB และหมุนเวียนเก็บสำรองสูงสุด 3 ไฟล์ (.log, .log.1, .log.2)
        file_handler = RotatingFileHandler(log_path, maxBytes=500000, backupCount=3, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("=== Tumor Segmentation Logger Initialized Successfully ===")
        
    except Exception as e:
        # แผนสำรอง: หากเครื่องปลายทางล็อกสิทธิ์โฟลเดอร์ (Write Access Denied) ให้ย้ายไปเซฟที่ Temp ของระบบ
        fallback_dir = slicer.app.temporaryPath
        log_path = os.path.join(fallback_dir, "tumor_segmentation_fallback.log")
        file_handler = RotatingFileHandler(log_path, maxBytes=500000, backupCount=3, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.warning(f"Cannot write log to module directory. Fallback to Slicer Temp: {e}")

    return logger

# เรียกใช้งาน Logger ระดับ Global ภายในไฟล์นี้
log = setup_module_logger()


#
# Tumor_Segmentation
#

class Tumor_Segmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Tumor SegmentationTest")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Registration")]
        self.parent.dependencies = []  
        self.parent.contributors = ["Noppanon Nobnop (Department of Biomedical Engineering Srinakharinwirot University.)"]  
        self.parent.helpText = _("""
3D Slicer extension for tumor segmentation in MRI using "Tumor_Segmentator" AI model.                                                
""")
        self.parent.acknowledgementText = _("""
This file was originally developed by Noppanon Nobnop (ImageLab, Srinakharinwirot University).
""")
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """Add data sets to Sample Data module."""
    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="Tumor_Segmentation",
        sampleName="Tumor_Segmentation1",
        thumbnailFileName=os.path.join(iconsPath, "Tumor_Segmentation1.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="Tumor_Segmentation1.nrrd",
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        nodeNames="Tumor_Segmentation1",
    )

    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        category="Tumor_Segmentation",
        sampleName="Tumor_Segmentation2",
        thumbnailFileName=os.path.join(iconsPath, "Tumor_Segmentation2.png"),
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="Tumor_Segmentation2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
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

        self._logic = None  # เปลี่ยนเป็น Private Variable เพื่อทำ Lazy Loading
        self._parameterNode = None
        self._parameterNodeGuiTag = None

        self.startSlice = None
        self.endSlice = None
        self.roiNode = None

    @property
    def logic(self):
        """เรียกใช้ Logic แบบ Lazy Loading เมื่อต้องการใช้งานจริงๆ เท่านั้น ป้องกันปัญหาข้ามเซสชัน"""
        if self._logic is None:
            self._logic = Tumor_SegmentationLogic()
        return self._logic

    def configureDependencies(self):
        """เช็คและดาวน์โหลดภายนอกผ่านการจัดเซสชันของระบบจัดการโมดูล"""
        import importlib
        required_packages = {
            "onnxruntime": "onnxruntime",
            "cv2": "opencv-python"
        }

        for module_name, pip_name in required_packages.items():
            try:
                importlib.import_module(module_name)
            except ImportError:
                log.info(f"--- ไม่พบ {module_name} กำลังดาวน์โหลดและติดตั้ง {pip_name}... ---")
                
                progress_dialog = slicer.util.createProgressDialog(
                    labelText=f"Installing {pip_name} for Tumor Segmentation. Please wait...",
                    maximum=0
                )
                slicer.app.processEvents()

                try:
                    slicer.util.pip_install(pip_name)
                    importlib.invalidate_caches()
                    importlib.import_module(module_name)
                    log.info(f"--- ติดตั้ง {pip_name} สำเร็จ! ---")
                except Exception as e:
                    log.error(f"Failed to install {pip_name}: {e}")
                
                progress_dialog.close()

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # ย้ายการทำงานมาไว้ที่จุดเริ่มต้นของ setup() เพื่อความปลอดภัยสูงสุดของหน่วยความจำเซสชัน
        self.configureDependencies()

        # Path setup
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Tumor_Segmentation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Segment Editor setup
        self.ui.roiSegmentEditor.setMRMLScene(slicer.mrmlScene)
        self.roiEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.ui.roiSegmentEditor.setMRMLSegmentEditorNode(self.roiEditorNode)

        # Connections
        self.ui.inputSelector.currentNodeChanged.connect(self.onInputChanged)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.show3DButton.connect("clicked(bool)", self.onShow3DButton)
        self.ui.createRoiButton.clicked.connect(self.onCreateRoi)
        self.ui.useRoiCheckBox.connect("toggled(bool)", self.onUseRoiToggled)
        self.ui.setStartSliceButton.connect("clicked(bool)", self.onSetStartSlice)
        self.ui.setEndSliceButton.connect("clicked(bool)", self.onSetEndSlice)

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

    def onSetStartSlice(self):
        lm = slicer.app.layoutManager()
        redWidget = lm.sliceWidget("Red")
        sliceLogic = redWidget.sliceLogic()
        sliceIndex = sliceLogic.GetSliceIndexFromOffset(sliceLogic.GetSliceOffset())
        self.startSlice = sliceIndex
        self.ui.startSliceLabel.setText(f"Start: {sliceIndex}")
        log.info(f"Start Slice กำหนดเป็น: {sliceIndex}")

    def onSetEndSlice(self):
        lm = slicer.app.layoutManager()
        redWidget = lm.sliceWidget("Red")
        sliceLogic = redWidget.sliceLogic()
        sliceIndex = sliceLogic.GetSliceIndexFromOffset(sliceLogic.GetSliceOffset())
        self.endSlice = sliceIndex
        self.ui.endSliceLabel.setText(f"End: {sliceIndex}")
        log.info(f"End Slice กำหนดเป็น: {sliceIndex}")

    def cleanup(self):
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None):
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputVolume:
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.enabled = False

    def onApplyButton(self):
        log.info("ผู้ใช้คลิก Apply Button เพื่อเริ่มกระบวนการ Segmentation")
        with slicer.util.tryWithErrorDisplay(_("Tumor segmentation failed."), waitCursor=True):
            inputNode = self.ui.inputSelector.currentNode()
            if not inputNode:
                raise RuntimeError("Please select input volume")

            vol = slicer.util.arrayFromVolume(inputNode)
            depth = vol.shape[0]

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

            roiNode = None
            if self.ui.useRoiCheckBox.checked:
                if self.roiNode is None:
                    raise RuntimeError("Please create ROI first")
                roiNode = self.roiNode

            # ส่งต่อการทำงานไปให้ตัว Logic
            segNode = self.logic.process(
                inputVolume=inputNode,
                outputVolume=None,
                startSlice=startSlice,
                endSlice=endSlice,
                roiNode=roiNode
            )

            self.logic._showAxialOnly(inputNode, segNode)
            self._currentSegNode = segNode
            self.ui.show3DButton.enabled = True
            log.info("กระบวนการ Segmentation ประสบความสำเร็จ")

    def onCreateRoi(self):
        inputNode = self.ui.inputSelector.currentNode()
        if not inputNode:
            slicer.util.errorDisplay("Please select input volume first")
            return
        if self.startSlice is None or self.endSlice is None:
            slicer.util.errorDisplay("Please set Start / End slice first")
            return

        self.roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "ROI_Segmentation")
        self.roiNode.CreateDefaultDisplayNodes()
        self.roiNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume=inputNode)
        self.roiNode.GetSegmentation().AddEmptySegment("ROI")

        self.ui.roiSegmentEditor.setSegmentationNode(self.roiNode)
        self.ui.roiSegmentEditor.setSourceVolumeNode(inputNode)
        slicer.util.infoDisplay("Draw ROI using Paint or Draw tool")
        log.info("สร้าง Node ROI เรียบร้อยแล้ว พร้อมให้ผู้ใช้วาดมาร์กเกอร์")

    def onShow3DButton(self):
        log.info("เรียกแสดงผลโมเดล 3D")
        segNode = getattr(self, "_currentSegNode", None)
        if not segNode:
            return
        displayNode = segNode.GetDisplayNode()
        if not displayNode:
            return
        displayNode.RemoveAllViewNodeIDs()
        displayNode.SetVisibility(True)
        displayNode.SetVisibility3D(True)

        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()


#
# Tumor_SegmentationLogic
#

class Tumor_SegmentationLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.session = None
        self.inputName = None
        self.modelH = 256
        self.modelW = 256

    def _ensureSession(self):
        if self.session is not None:
            return
            
        import onnxruntime as ort
        
        self.modelPath = os.path.join(os.path.dirname(__file__), "Resources", "Models", "unet.onnx")
        log.info(f"กำลังโหลดโมเดล AI จากโฟลเดอร์ทรัพยากร: {self.modelPath}")
        
        self.session = ort.InferenceSession(self.modelPath, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
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

        redWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(volumeNode.GetID())
        greenWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(None)
        yellowWidget.sliceLogic().GetSliceCompositeNode().SetBackgroundVolumeID(None)

        displayNode = segNode.GetDisplayNode()
        if not displayNode:
            displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationDisplayNode")
            segNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        displayNode.SetVisibility(True)
        displayNode.SetVisibility2D(True)
        displayNode.SetVisibility3D(False)
        displayNode.SetVisibility2DFill(True)
        displayNode.SetVisibility2DOutline(True)
        displayNode.AddViewNodeID(redSliceNode.GetID())

        redSliceNode.SetOrientationToAxial()
        redWidget.sliceLogic().FitSliceToAll()

    def rasToIjk(self, volumeNode, sliceOffset):
        ras = [0, 0, sliceOffset, 1]
        ijk = [0, 0, 0, 0]
        volumeNode.GetRASToIJKMatrix().MultiplyPoint(ras, ijk)
        return ijk

    def _resliceToAxial(self, inputVolume):
        axialVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "AxialReslicedTemp")
        params = {
            "inputVolume": inputVolume.GetID(),
            "outputVolume": axialVolume.GetID(),
            "orientation": "Axial",
            "interpolationType": "linear"
        }
        slicer.cli.runSync(slicer.modules.resamplescalarvolume, None, params)
        axialVolume.GetImageData().Modified()
        axialVolume.Modified()
        if axialVolume.GetImageData() is None:
            raise RuntimeError("Axial reslicing failed: output volume has no image data")
        return axialVolume

    def getParameterNode(self):
        return Tumor_SegmentationParameterNode(super().getParameterNode())

    def process(self, inputVolume, outputVolume=None, startSlice=0, endSlice=None, roiNode=None):
        import cv2
        log.info(f"เริ่มการทำงานในระบบ Logic: สไลซ์เริ่มต้นที่ {startSlice} ถึง {endSlice}")
        
        self._ensureSession()

        vol = slicer.util.arrayFromVolume(inputVolume)
        depth, height, width = vol.shape

        if endSlice is None:
            endSlice = depth - 1

        outputMask = np.zeros_like(vol, dtype=np.uint8)
        roiMask = None
        if roiNode is not None:
            roiMask = slicer.util.arrayFromSegmentBinaryLabelmap(roiNode, "ROI", inputVolume)

        for i in range(startSlice, endSlice + 1):
            sliceImg = vol[i]
            img = cv2.resize(sliceImg, (self.modelW, self.modelH))
            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)

            outputs = self.session.run(None, {self.inputName: img})
            raw_result = np.squeeze(outputs[0])

            exp_x = np.exp(raw_result - np.max(raw_result, axis=0))
            prob_map = exp_x / exp_x.sum(axis=0)

            ch0_tumor = prob_map[0, :, :]
            ch1_background = prob_map[1, :, :]
            pred = (ch0_tumor > ch1_background).astype(np.uint8)
            pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)

            if roiMask is not None:
                roiSlice = roiMask[i]
                roiSlice = (roiSlice > 0).astype(np.uint8)
                if roiSlice.shape != pred.shape:
                    roiSlice = cv2.resize(roiSlice, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred = pred * roiSlice

            outputMask[i] = pred

        if outputVolume is None:
            outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "TumorMaskVolume")
            outputVolume.CreateDefaultDisplayNodes()

        slicer.util.updateVolumeFromArray(outputVolume, outputMask)

        labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "TumorLabelMap")
        slicer.util.updateVolumeFromArray(labelmapNode, outputMask)
        labelmapNode.SetSpacing(inputVolume.GetSpacing())
        labelmapNode.SetOrigin(inputVolume.GetOrigin())

        ijkToRAS = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRAS)
        labelmapNode.SetIJKToRASMatrix(ijkToRAS)

        segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "TumorSegmentation")
        segNode.CreateDefaultDisplayNodes()
        segNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapNode, segNode)

        seg = segNode.GetSegmentation()
        segmentID = seg.GetNthSegmentID(0)
        segment = seg.GetSegment(segmentID)
        segment.SetColor(1.0, 0.0, 0.0)

        slicer.mrmlScene.RemoveNode(labelmapNode)
        
        log.info("การคำนวณและอัปเดต Volume หน้าจอเสร็จสิ้น")
        return segNode