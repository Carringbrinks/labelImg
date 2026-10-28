#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pure ONNX-Runtime object detector for labelImg pre-annotation.

Runs an Ultralytics-style exported detection ONNX model (input: [1, 3, 640, 640]
RGB float32 in [0, 1] with letterbox padding color 114; output: [1, 4+nc, N]
with cxcywh boxes already DFL-decoded and sigmoided class scores) without any
dependency on ultralytics / torch / opencv.

Image preprocessing is done with QImage (already available through PyQt5) so
the only runtime requirements are onnxruntime and numpy.
"""
import os
import threading
import time
import hashlib

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPainter, QPen

LETTERBOX_COLOR = 114

# Vivid palette for detection results (per class index), solid line color
# with a light fill so pre-annotations stand out from manual boxes.
_DETECT_PALETTE = [(255, 90, 0), (0, 200, 255), (255, 200, 0),
                   (120, 255, 60), (255, 60, 180), (140, 120, 255),
                   (255, 140, 60), (0, 255, 160)]
_DETECT_LINE_ALPHA = 255
_DETECT_FILL_ALPHA = 70
_DETECT_LINE_WIDTH = 1.5  # multiplier on the default 2px shape outline


class Detection(object):
    """One detection: label plus pixel box on the original image."""

    __slots__ = ('label', 'xmin', 'ymin', 'xmax', 'ymax', 'conf')

    def __init__(self, label, xmin, ymin, xmax, ymax, conf):
        self.label = label
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)
        self.conf = float(conf)

    @staticmethod
    def colorForLabel(label, classes=None):
        """Vivid line color for a detection label (palette by class index,
        hash fallback for unknown labels)."""
        if classes and label in classes:
            r, g, b = _DETECT_PALETTE[classes.index(label) % len(_DETECT_PALETTE)]
        else:
            h = int(hashlib.sha256(label.encode('utf-8')).hexdigest(), 16)
            r, g, b = h % 255, (h // 255) % 255, (h // 65025) % 255
        return (r, g, b)

    def asShape(self, classes=None):
        """Return the (label, points, line_color, fill_color, difficult,
        line_width) tuple expected by MainWindow.loadLabels()."""
        points = [(self.xmin, self.ymin), (self.xmax, self.ymin),
                  (self.xmax, self.ymax), (self.xmin, self.ymax)]
        r, g, b = self.colorForLabel(self.label, classes)
        return (self.label, points,
                (r, g, b, _DETECT_LINE_ALPHA),
                (r, g, b, _DETECT_FILL_ALPHA),
                False, _DETECT_LINE_WIDTH)


class OnnxDetector(object):
    """ONNX-Runtime inference wrapper.

    The session is created lazily by load() (idempotent) and is meant to be
    called from a single worker thread, which is how DetectionWorker uses it.
    """

    def __init__(self, model_path, classes=None, conf_thres=0.25,
                 iou_thres=0.45, device='CPU'):
        self.model_path = model_path
        self.classes = list(classes) if classes else []
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device or 'CPU'
        self.session = None
        self.input_name = None
        self.imgsz = 640
        self.nc = len(self.classes)

    # ------------------------------------------------------------------ utils

    def load(self):
        """Create the inference session (idempotent). Call from worker thread."""
        if self.session is not None:
            return
        import onnxruntime as ort
        if not os.path.exists(self.model_path):
            raise IOError(u'ONNX模型文件不存在: %s' % self.model_path)
        providers = []
        if self.device.upper() == 'CUDA' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if len(inp.shape) == 4 and inp.shape[1] == 3 and inp.shape[2]:
            self.imgsz = int(inp.shape[2])

    # ----------------------------------------------------------- preprocessing

    @staticmethod
    def qimageToRgbArray(qimage):
        """Return an (h, w, 3) uint8 numpy array copy of a QImage."""
        import numpy as np
        img = qimage.convertToFormat(QImage.Format_RGB888)
        data = img.constBits().asstring(img.sizeInBytes())
        arr = np.frombuffer(data, np.uint8).reshape(
            img.height(), img.bytesPerLine())[:, :img.width() * 3]
        return arr.reshape(img.height(), img.width(), 3)

    def letterboxBlob(self, qimage):
        """Letterbox-resize a QImage into the model input blob.

        Returns (blob [1,3,imgsz,imgsz] float32, ratio, (left, top)).
        """
        import numpy as np
        w0, h0 = qimage.width(), qimage.height()
        s = self.imgsz
        ratio = min(float(s) / h0, float(s) / w0)
        nw, nh = int(round(w0 * ratio)), int(round(h0 * ratio))
        scaled = qimage.scaled(nw, nh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        canvas = QImage(s, s, QImage.Format_RGB888)
        canvas.fill(QColor(LETTERBOX_COLOR, LETTERBOX_COLOR, LETTERBOX_COLOR))
        painter = QPainter(canvas)
        left, top = (s - nw) // 2, (s - nh) // 2
        painter.drawImage(left, top, scaled)
        painter.end()
        arr = self.qimageToRgbArray(canvas)
        blob = arr.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return blob, ratio, (left, top)

    # ---------------------------------------------------------- postprocessing

    @staticmethod
    def nms(boxes, scores, iou_thres):
        """Numpy NMS over xyxy boxes of a SINGLE class. Returns kept indices."""
        import numpy as np
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (
                (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                + (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
                - inter + 1e-9)
            order = order[1:][iou <= iou_thres]
        return keep

    def postprocess(self, output, ratio, pad, w0, h0):
        """Convert raw [1, 4+nc, N] output into a list of Detection."""
        import numpy as np
        out = output[0].T  # (N, 4+nc)
        boxes = out[:, :4]  # cx, cy, w, h in letterboxed space
        scores = out[:, 4:]
        cls_ids = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        mask = confs > self.conf_thres
        boxes, cls_ids, confs = boxes[mask], cls_ids[mask], confs[mask]
        if boxes.size == 0:
            return []
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2,
                         boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2,
                         boxes[:, 1] + boxes[:, 3] / 2], axis=1)
        # Class-wise NMS: boxes of different classes may legitimately overlap
        # or contain each other (e.g. a shelf box containing book spines), so
        # IoU suppression is applied within each class only. This matches the
        # behaviour the model was trained/validated with.
        keep = []
        for cls_id in np.unique(cls_ids):
            idx = np.where(cls_ids == cls_id)[0]
            keep.extend(idx[self.nms(xyxy[idx], confs[idx], self.iou_thres)])
        xyxy, cls_ids, confs = xyxy[keep], cls_ids[keep], confs[keep]
        # Undo letterbox and clip to the original image.
        left, top = pad
        xyxy = (xyxy - np.array([left, top, left, top], dtype=np.float32)) / ratio
        xyxy = np.clip(xyxy, 0.0, np.array([w0, h0, w0, h0], dtype=np.float32))
        detections = []
        for i in range(len(xyxy)):
            label = self.classes[int(cls_ids[i])] \
                if int(cls_ids[i]) < len(self.classes) else 'cls%d' % cls_ids[i]
            detections.append(Detection(label, xyxy[i, 0], xyxy[i, 1],
                                        xyxy[i, 2], xyxy[i, 3], confs[i]))
        detections.sort(key=lambda d: -d.conf)
        return detections

    # -------------------------------------------------------------- detection

    def detectImageData(self, imageData):
        """Run detection on raw image bytes. Returns (detections, elapsed_ms)."""
        t0 = time.time()
        qimage = QImage.fromData(imageData)
        if qimage.isNull():
            raise ValueError(u'图片数据无法解码')
        w0, h0 = qimage.width(), qimage.height()
        blob, ratio, pad = self.letterboxBlob(qimage)
        output = self.session.run(None, {self.input_name: blob})[0]
        detections = self.postprocess(output, ratio, pad, w0, h0)
        return detections, (time.time() - t0) * 1000.0

    def detectFile(self, filePath):
        with open(filePath, 'rb') as f:
            return self.detectImageData(f.read())


# ------------------------------------------------------------ YOLO txt export

def writeYoloTxt(txtPath, detections, imgW, imgH, classList):
    """Write detections in YOLO format using the model's class order.

    Unlike YOLOWriter (which assigns class indices by first appearance), the
    index here is the position of the label in classList, so the file is
    directly usable for training with the same class order as the model.
    """
    with open(txtPath, 'w', encoding='utf-8') as f:
        for d in detections:
            if d.label in classList:
                clsIdx = classList.index(d.label)
            elif d.label.startswith('cls') and d.label[3:].isdigit():
                clsIdx = int(d.label[3:])
            else:
                clsIdx = 0
            xc = (d.xmin + d.xmax) / 2.0 / imgW
            yc = (d.ymin + d.ymax) / 2.0 / imgH
            w = (d.xmax - d.xmin) / imgW
            h = (d.ymax - d.ymin) / imgH
            f.write('%d %.6f %.6f %.6f %.6f\n' % (clsIdx, xc, yc, w, h))


def writeClassesTxt(classesPath, classList):
    with open(classesPath, 'w', encoding='utf-8') as f:
        for c in classList:
            f.write(c + '\n')


_PREVIEW_COLORS = [(255, 80, 80), (80, 160, 255), (255, 200, 60),
                   (80, 255, 120), (255, 120, 220), (120, 220, 255)]


def savePreviewImage(imageData, detections, classList, outPath):
    """Draw detection boxes on the image and save it as jpg."""
    qimage = QImage.fromData(imageData)
    if qimage.isNull():
        return False
    painter = QPainter(qimage)
    for d in detections:
        clsIdx = classList.index(d.label) if d.label in classList else 0
        r, g, b = _PREVIEW_COLORS[clsIdx % len(_PREVIEW_COLORS)]
        pen = QPen(QColor(r, g, b), 3)
        painter.setPen(pen)
        painter.drawRect(int(d.xmin), int(d.ymin),
                         int(d.xmax - d.xmin), int(d.ymax - d.ymin))
        painter.drawText(int(d.xmin), max(12, int(d.ymin) - 4),
                         u'%s %.2f' % (d.label, d.conf))
    painter.end()
    return qimage.save(outPath, 'jpg', 90)


# ------------------------------------------------------------------- threading

class DetectionWorker(QObject):
    """Detector worker living in its own thread (canonical moveToThread pattern).

    All slots execute sequentially in the worker thread, so queued requests
    never race and the onnxruntime session is reused safely across jobs.
    """

    resultReady = pyqtSignal(object)    # single-image payload
    progress = pyqtSignal(int, int)     # batch: (done, total)
    batchFinished = pyqtSignal(object)  # batch payload
    errorOccurred = pyqtSignal(str)

    def __init__(self):
        super(DetectionWorker, self).__init__()
        self.detector = None
        self._cancel = threading.Event()

    @pyqtSlot(object)
    def setDetector(self, detector):
        self.detector = detector

    def cancel(self):
        """Thread-safe cancel request (plain Event.set, called from any thread)."""
        self._cancel.set()

    @pyqtSlot(str, object)
    def detectSingle(self, filePath, imageData):
        self._cancel.clear()
        try:
            self.detector.load()
            detections, elapsed = self.detector.detectImageData(imageData)
            self.resultReady.emit({
                'mode': 'single',
                'filePath': filePath,
                'detections': detections,
                'elapsed_ms': elapsed,
            })
        except Exception as e:
            self.errorOccurred.emit(str(e))

    @pyqtSlot(object, str, bool, bool)
    def detectBatch(self, imagePaths, outDir, saveTxt, savePreview):
        self._cancel.clear()
        try:
            self.detector.load()
        except Exception as e:
            self.errorOccurred.emit(str(e))
            return
        payload = {'mode': 'batch', 'outDir': outDir,
                   'total': len(imagePaths), 'done': 0,
                   'failed': [], 'boxCount': 0, 'canceled': False}
        try:
            classList = self.detector.classes
            if saveTxt and classList:
                writeClassesTxt(os.path.join(outDir, 'classes.txt'), classList)
            if savePreview:
                previewDir = os.path.join(outDir, 'preview')
                if not os.path.isdir(previewDir):
                    os.makedirs(previewDir)
            for i, path in enumerate(imagePaths):
                if self._cancel.is_set():
                    payload['canceled'] = True
                    break
                try:
                    with open(path, 'rb') as f:
                        imageData = f.read()
                    qimage = QImage.fromData(imageData)
                    if qimage.isNull():
                        raise ValueError(u'无法解码图片')
                    detections, _ = self.detector.detectImageData(imageData)
                    basename = os.path.splitext(os.path.basename(path))[0]
                    if saveTxt:
                        writeYoloTxt(os.path.join(outDir, basename + '.txt'),
                                     detections, qimage.width(), qimage.height(),
                                     classList)
                    if savePreview:
                        savePreviewImage(imageData, detections, classList,
                                         os.path.join(previewDir, basename + '.jpg'))
                    payload['boxCount'] += len(detections)
                except Exception as e:
                    payload['failed'].append('%s: %s' % (os.path.basename(path), e))
                payload['done'] = i + 1
                self.progress.emit(payload['done'], payload['total'])
            self.batchFinished.emit(payload)
        except Exception as e:
            self.errorOccurred.emit(str(e))


class DetectionManager(QObject):
    """Owns the worker thread and dispatches detection requests to it.

    Requests are sent as queued signals, so they run strictly one after
    another in the worker thread (no QThread restart races) and the
    session is created once and reused.
    """

    requestSetDetector = pyqtSignal(object)
    requestSingle = pyqtSignal(str, object)
    requestBatch = pyqtSignal(object, str, bool, bool)

    def __init__(self, parent=None):
        super(DetectionManager, self).__init__(parent)
        self.worker = DetectionWorker()
        self._thread = QThread(self)
        self.worker.moveToThread(self._thread)
        self.requestSetDetector.connect(self.worker.setDetector)
        self.requestSingle.connect(self.worker.detectSingle)
        self.requestBatch.connect(self.worker.detectBatch)
        self._thread.start()

    def setDetector(self, detector):
        self.requestSetDetector.emit(detector)

    def runSingle(self, filePath, imageData):
        self.requestSingle.emit(filePath, imageData)

    def runBatch(self, imagePaths, outDir, saveTxt=True, savePreview=False):
        self.requestBatch.emit(imagePaths, outDir, saveTxt, savePreview)

    def cancel(self):
        self.worker.cancel()

    def shutdown(self):
        self.worker.cancel()
        self._thread.quit()
        self._thread.wait(5000)
