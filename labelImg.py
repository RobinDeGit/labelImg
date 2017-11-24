#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import os.path
import re
import sys
import subprocess

from functools import partial
from collections import defaultdict

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import resources
# Add internal libs
from libs.constants import *
from libs.lib import struct, newAction, newIcon, addActions, fmtShortcut, generateColorByText
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.ustr import ustr
from libs.version import __version__

__appname__ = 'labelImg'

# Utility functions and classes.




if sys.byteorder == 'little':
    _bgra = (0, 1, 2, 3)
else:
    _bgra = (3, 2, 1, 0)


try:
    _basestring = basestring
except NameError:
    # 'basestring' undefined, must be Python 3
    _basestring = str

def _normalize255(array, normalize, clip = (0, 255)):
    if normalize:
        if normalize is True:
            normalize = array.min(), array.max()
            if clip == (0, 255):
                clip = None
        elif np.isscalar(normalize):
            normalize = (0, normalize)

        nmin, nmax = normalize

        if nmin:
            array = array - nmin

        if nmax != nmin:
            scale = 255. / (nmax - nmin)
            if scale != 1.0:
                array = array * scale

    if clip:
        low, high = clip
        np.clip(array, low, high, array)

    return array

    
def rgb2qimage(rgb):
    """Convert the 3D numpy array `rgb` into a 32-bit QImage.  `rgb` must
    have three dimensions with the vertical, horizontal and RGB image axes.
    
    ATTENTION: This QImage carries an attribute `ndimage` with a
    reference to the underlying numpy array that holds the data. On
    Windows, the conversion into a QPixmap does not copy the data, so
    that you have to take care that the QImage does not get garbage
    collected (otherwise PyQt will throw away the wrapper, effectively
    freeing the underlying memory - boom!)."""
    if len(rgb.shape) != 3:
        raise ValueError("rgb2QImage can only convert 3D arrays")
    if rgb.shape[2] not in (3, 4):
        raise ValueError("rgb2QImage can expects the last dimension to contain exactly three (R,G,B) or four (R,G,B,A) channels")

    h, w, channels = rgb.shape

    # Qt expects 32bit BGRA data for color images:
    bgra = numpy.empty((h, w, 4), numpy.uint8, 'C')
    bgra[...,0] = rgb[...,2]
    bgra[...,1] = rgb[...,1]
    bgra[...,2] = rgb[...,0]
    if rgb.shape[2] == 3:
        bgra[...,3].fill(255)
        fmt = QtGui.QImage.Format_RGB32
    else:
        bgra[...,3] = rgb[...,3]
        fmt = QtGui.QImage.Format_ARGB32

    result = QtGui.QImage(bgra.data, w, h, fmt)
    result.ndarray = bgra
    return result



def PyQt_data(image):
    # PyQt4/PyQt5's QImage.bits() returns a sip.voidptr that supports
    # conversion to string via asstring(size) or getting its base
    # address via int(...):
    return (int(image.bits()), False)

def _re_buffer_address_match(buf_repr):
    import re
    _re_buffer_address = re.compile('<read-write buffer ptr 0x([0-9a-fA-F]*),')
    global _re_buffer_address_match
    _re_buffer_address_match = _re_buffer_address.match
    return _re_buffer_address_match(buf_repr)

def PySide_data(image):
    # PySide's QImage.bits() returns a buffer object like this:
    # <read-write buffer ptr 0x7fc3f4821600, size 76800 at 0x111269570>
    ma = _re_buffer_address_match(repr(image.bits()))
    assert ma, 'could not parse address from %r' % (image.bits(), )
    return (int(ma.group(1), 16), False)

def direct_buffer_data(image):
    return image.bits()



validFormats_8bit = [getattr(QImage, name) for name in ('Format_Indexed8', 'Format_Grayscale8') if name in dir(QImage)]
validFormats_32bit = (QImage.Format_RGB32, QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied)

def qimageview(image):
    if not isinstance(image, QImage):
        raise TypeError("image argument must be a QImage instance")

    shape = image.height(), image.width()
    strides0 = image.bytesPerLine()

    format = image.format()
    if format in validFormats_8bit:
        dtype = "|u1"
        strides1 = 1
    elif format in validFormats_32bit:
        dtype = "|u4"
        strides1 = 4
    elif format == QImage.Format_Invalid:
        raise ValueError("qimageview got invalid QImage")
    else:
        raise ValueError("qimageview can only handle 8- or 32-bit QImages (format was %r)" % format)

    image.__array_interface__ = {
        'shape': shape,
        'typestr': dtype,
        'data': PyQt_data(image),
        'strides': (strides0, strides1),
        'version': 3,
    }

    result = np.asarray(image)
    del image.__array_interface__
    return result



def _qimage_or_filename_view(qimage):
    if isinstance(qimage, _basestring):
        qimage = _qt.QImage(qimage)
    return qimageview(qimage)

def raw_view(qimage):
    """Returns raw 2D view of the given QImage_'s memory.  The result
    will be a 2-dimensional numpy.ndarray with an appropriately sized
    integral dtype.  (This function is not intented to be used
    directly, but used internally by the other -- more convenient --
    view creation functions.)

    :param qimage: image whose memory shall be accessed via NumPy
    :type qimage: QImage_
    :rtype: numpy.ndarray_ with shape (height, width)"""
    return _qimage_or_filename_view(qimage)


def byte_view(qimage, byteorder = 'little'):
    """Returns raw 3D view of the given QImage_'s memory.  This will
    always be a 3-dimensional numpy.ndarray with dtype numpy.uint8.
    
    Note that for 32-bit images, the last dimension will be in the
    [B,G,R,A] order (if little endian) due to QImage_'s memory layout
    (the alpha channel will be present for Format_RGB32 images, too).

    For 8-bit (indexed) images, the array will still be 3-dimensional,
    i.e. shape will be (height, width, 1).

    The order of channels in the last axis depends on the `byteorder`,
    which defaults to 'little', i.e. BGRA order.  You may set the
    argument `byteorder` to 'big' to get ARGB, or use None which means
    sys.byteorder here, i.e. return native order for the machine the
    code is running on.

    For your convenience, `qimage` may also be a filename, see
    `Loading and Saving Images`_ in the documentation.

    :param qimage: image whose memory shall be accessed via NumPy
    :type qimage: QImage_
    :param byteorder: specify order of channels in last axis
    :rtype: numpy.ndarray_ with shape (height, width, 1 or 4) and dtype uint8"""
    raw = _qimage_or_filename_view(qimage)
    result = raw.view(np.uint8).reshape(raw.shape + (-1, ))
    if byteorder and byteorder != sys.byteorder:
        result = result[...,::-1]
    return result


def rgb_view(qimage, byteorder = 'big'):
    """Returns RGB view of a given 32-bit color QImage_'s memory.
    Similarly to byte_view(), the result is a 3D numpy.uint8 array,
    but reduced to the rgb dimensions (without alpha), and reordered
    (using negative strides in the last dimension) to have the usual
    [R,G,B] order.  The image must have 32 bit pixel size, i.e. be
    RGB32, ARGB32, or ARGB32_Premultiplied.  (Note that in the latter
    case, the values are of course premultiplied with alpha.)

    The order of channels in the last axis depends on the `byteorder`,
    which defaults to 'big', i.e. RGB order.  You may set the argument
    `byteorder` to 'little' to get BGR, or use None which means
    sys.byteorder here, i.e. return native order for the machine the
    code is running on.

    For your convenience, `qimage` may also be a filename, see
    `Loading and Saving Images`_ in the documentation.

    :param qimage: image whose memory shall be accessed via NumPy
    :type qimage: QImage_ with 32-bit pixel type
    :param byteorder: specify order of channels in last axis
    :rtype: numpy.ndarray_ with shape (height, width, 3) and dtype uint8"""
    if byteorder is None:
        byteorder = sys.byteorder
    bytes = byte_view(qimage, byteorder)
    if bytes.shape[2] != 4:
        raise ValueError("For rgb_view, the image must have 32 bit pixel size (use RGB32, ARGB32, or ARGB32_Premultiplied)")

    if byteorder == 'little':
        return bytes[...,:3] # strip A off BGRA
    else:
        return bytes[...,1:] # strip A off ARGB



def alpha_view(qimage):
    """Returns alpha view of a given 32-bit color QImage_'s memory.
    The result is a 2D numpy.uint8 array, equivalent to
    byte_view(qimage)[...,3].  The image must have 32 bit pixel size,
    i.e. be RGB32, ARGB32, or ARGB32_Premultiplied.  Note that it is
    not enforced that the given qimage has a format that actually
    *uses* the alpha channel -- for Format_RGB32, the alpha channel
    usually contains 255 everywhere.

    For your convenience, `qimage` may also be a filename, see
    `Loading and Saving Images`_ in the documentation.

    :param qimage: image whose memory shall be accessed via NumPy
    :type qimage: QImage_ with 32-bit pixel type
    :rtype: numpy.ndarray_ with shape (height, width) and dtype uint8"""
    bytes = byte_view(qimage, byteorder = None)
    if bytes.shape[2] != 4:
        raise ValueError("For alpha_view, the image must have 32 bit pixel size (use RGB32, ARGB32, or ARGB32_Premultiplied)")
    return bytes[...,_bgra[3]]



def byte_view(qimage, byteorder = 'little'):
    """Returns raw 3D view of the given QImage_'s memory.  This will
    always be a 3-dimensional numpy.ndarray with dtype numpy.uint8.
    
    Note that for 32-bit images, the last dimension will be in the
    [B,G,R,A] order (if little endian) due to QImage_'s memory layout
    (the alpha channel will be present for Format_RGB32 images, too).

    For 8-bit (indexed) images, the array will still be 3-dimensional,
    i.e. shape will be (height, width, 1).

    The order of channels in the last axis depends on the `byteorder`,
    which defaults to 'little', i.e. BGRA order.  You may set the
    argument `byteorder` to 'big' to get ARGB, or use None which means
    sys.byteorder here, i.e. return native order for the machine the
    code is running on.

    For your convenience, `qimage` may also be a filename, see
    `Loading and Saving Images`_ in the documentation.

    :param qimage: image whose memory shall be accessed via NumPy
    :type qimage: QImage_
    :param byteorder: specify order of channels in last axis
    :rtype: numpy.ndarray_ with shape (height, width, 1 or 4) and dtype uint8"""
    raw = _qimage_or_filename_view(qimage)
    result = raw.view(np.uint8).reshape(raw.shape + (-1, ))
    if byteorder and byteorder != sys.byteorder:
        result = result[...,::-1]
    return result


def array2qimage(array, normalize = False):
    """Convert a 2D or 3D numpy array into a 32-bit QImage_.  The
    first dimension represents the vertical image axis; the optional
    third dimension is supposed to contain 1-4 channels:

    ========= ===================
    #channels interpretation
    ========= ===================
            1 scalar/gray
            2 scalar/gray + alpha
            3 RGB
            4 RGB + alpha
    ========= ===================

    Scalar data will be converted into corresponding gray RGB triples;
    if you want to convert to an (indexed) 8-bit image instead, use
    `gray2qimage` (which cannot support an alpha channel though).

    The parameter `normalize` can be used to normalize an image's
    value range to 0..255:

    `normalize` = (nmin, nmax):
      scale & clip image values from nmin..nmax to 0..255

    `normalize` = nmax:
      lets nmin default to zero, i.e. scale & clip the range 0..nmax
      to 0..255

    `normalize` = True:
      scale image values to 0..255 (same as passing (array.min(),
      array.max()))

    If `array` contains masked values, the corresponding pixels will
    be transparent in the result.  Thus, the result will be of
    QImage.Format_ARGB32 if the input already contains an alpha
    channel (i.e. has shape (H,W,4)) or if there are masked pixels,
    and QImage.Format_RGB32 otherwise.

    :param array: image data which should be converted (copied) into a QImage_
    :type array: 2D or 3D numpy.ndarray_ or `numpy.ma.array <masked arrays>`_
    :param normalize: normalization parameter (see above, default: no value changing)
    :type normalize: bool, scalar, or pair
    :rtype: QImage_ with RGB32 or ARGB32 format"""
    if np.ndim(array) == 2:
        array = array[...,None]
    elif np.ndim(array) != 3:
        raise ValueError("array2qimage can only convert 2D or 3D arrays (got %d dimensions)" % np.ndim(array))
    if array.shape[2] not in (1, 2, 3, 4):
        raise ValueError("array2qimage expects the last dimension to contain exactly one (scalar/gray), two (gray+alpha), three (R,G,B), or four (R,G,B,A) channels")

    h, w, channels = array.shape

    hasAlpha = np.ma.is_masked(array) or channels in (2, 4)
    fmt = QImage.Format_ARGB32 if hasAlpha else QImage.Format_RGB32

    result = QImage(w, h, fmt)

    array = _normalize255(array, normalize)

    if channels >= 3:
        rgb_view(result)[:] = array[...,:3]
    else:
        rgb_view(result)[:] = array[...,:1] # scalar data

    alpha = alpha_view(result)

    if channels in (2, 4):
        alpha[:] = array[...,-1]
    else:
        alpha[:] = 255

    if np.ma.is_masked(array):
        alpha[:]  *= np.logical_not(np.any(array.mask, axis = -1))

    return result




import cv2
import numpy as np

def pseudoRGB (img, method = "clahe", visualize = False):
    if method not in ["clahe"]:
        exit ("Pseudo RGB method " + str(method) + " is unknown.")

    conversionFactor = 256 
    if img.dtype == np.uint8:
        conversionFactor  = 1
        method = 'clahe'
    
    if method == "clahe":
        if img.shape[0] > 512 or img.shape[1] > 512:
            factor = max(img.shape[0], img.shape[1])/512
            clipfactor = 256 
        else:
            factor = 1
            clipfactor = 1
            
        clahe = cv2.createCLAHE(clipLimit=8.0*clipfactor, tileGridSize=(int(2*factor),int(2*factor)))
        red = (clahe.apply(img)/conversionFactor ).astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=2.0*clipfactor, tileGridSize=(int(8*factor),int(8*factor)))
        blue = (clahe.apply(img)/conversionFactor ).astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=4.0*clipfactor, tileGridSize=(int(4*factor),int(4*factor)))
        green = (clahe.apply(img)/conversionFactor ).astype('uint8')
        
        img =  cv2.merge((blue, green, red))
        
        if visualize == True:
            cv2.imshow('image512',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for i in range (1,5):
                cv2.waitKey(1)

    return img




def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


# PyQt5: TypeError: unhashable type: 'QListWidgetItem'
class HashableQListWidgetItem(QListWidgetItem):

    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Save as Pascal voc xml
        self.defaultSaveDir = None
        self.usingPascalVocFormat = True
        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = "firefox"
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(u'Use default label')
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(u'difficult')
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefaultLabelContainer)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        self.dock = QDockWidget(u'Box Labels', self)
        self.dock.setObjectName(u'Labels')
        self.dock.setWidget(labelListContainer)

        # Tzutalin 20160906 : Add file list and dock to move faster
        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(u'File List', self)
        self.filedock.setObjectName(u'Files')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas()
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        # Tzutalin 20160906 : Add file list and dock to move faster
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.dockFeatures = QDockWidget.DockWidgetClosable\
            | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action('&Quit', self.close,
                      'Ctrl+Q', 'quit', u'Quit application')

        open = action('&Open', self.openFile,
                      'Ctrl+O', 'open', u'Open image or label file')

        opendir = action('&Open Dir', self.openDirDialog,
                         'Ctrl+u', 'open', u'Open Dir')

        changeSavedir = action('&Change Save Dir', self.changeSavedirDialog,
                               'Ctrl+r', 'open', u'Change default saved Annotation dir')

        openAnnotation = action('&Open Annotation', self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', u'Open Annotation')

        openNextImg = action('&Next Image', self.openNextImg,
                             'd', 'next', u'Open Next')

        openPrevImg = action('&Prev Image', self.openPrevImg,
                             'a', 'prev', u'Open Prev')

        verify = action('&Verify Image', self.verifyImg,
                        'space', 'verify', u'Verify Image')


        scaleLeft = action('&Scale Left', self.scaleLeft,
                        '4', 'verify', u'Scale Left')
        scaleUp = action('&Scale Up', self.scaleUp,
                        '8', 'verify', u'Scale Up')
        scaleDown = action('&Scale Down', self.scaleDown,
                        '2', 'verify', u'Scale Down')
        scaleRight = action('&Scale Right', self.scaleRight,
                        '6', 'verify', u'Scale Right')

        
        resetAll = action('&ResetAll', self.resetAll, None, 'resetall', u'Reset all')

        upscaleAll = action('&Upscale All', self.upscaleAll,
                        '+', 'verify', u'Upscall All')
        downscaleAll = action('&Downscale All', self.downscaleAll,
                        '-', 'verify', u'Downscale All')
        moveLeft = action('&Move Left', self.moveLeft,
                        'Left', 'verify', u'Move Left')
        moveUp = action('&Move Up', self.moveUp,
                        'Up', 'verify', u'Move Up')
        moveDown = action('&Move Down', self.moveDown,
                        'Down', 'verify', u'Move Down')
        moveRight = action('&Move Right', self.moveRight,
                        'Right', 'verify', u'Move Right')
        flipHorizontal = action('&Shift horizontally', self.flipHorizontal,
                        'Shift+H', 'verify', u'Shift horizontally')

        save = action('&Save', self.saveFile,
                      'Ctrl+S', 'save', u'Save labels to file', enabled=False)
        saveAs = action('&Save As', self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', u'Save labels to a different file',
                        enabled=False)
        loadTemplate = action('&Load Template', self.loadTemplate,
                        'Ctrl+T', 'verify', u'Load Template')
        close = action('&Close', self.closeFile,
                       'Ctrl+W', 'close', u'Close current file')
        color1 = action('Box &Line Color', self.chooseColor1,
                        'Ctrl+L', 'color_line', u'Choose Box line color')
        color2 = action('Box &Fill Color', self.chooseColor2,
                        'Ctrl+Shift+L', 'color', u'Choose Box fill color')

        createMode = action('Create\nRectBox', self.setCreateMode,
                            'Ctrl+N', 'new', u'Start drawing Boxs', enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action('Create\nRectBox', self.createShape,
                        'w', 'new', u'Draw a new Box', enabled=False)
        delete = action('Delete\nRectBox', self.deleteSelectedShape,
                        'Delete', 'delete', u'Delete', enabled=False)
        copy = action('&Duplicate\nRectBox', self.copySelectedShape,
                      'Ctrl+D', 'copy', u'Create a duplicate of the selected Box',
                      enabled=False)

        advancedMode = action('&Advanced Mode', self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', u'Switch to advanced mode',
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', u'Hide all Boxs',
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', u'Show all Boxs',
                         enabled=False)

        help = action('&Tutorial', self.showTutorialDialog, None, 'help', u'Show demos')
        showInfo = action('&Information', self.showInfoDialog, None, 'help', u'Information')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In', partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', u'Increase zoom level', enabled=False)
        zoomOut = action('&Zoom Out', partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', u'Decrease zoom level', enabled=False)
        zoomOrg = action('&Original size', partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', u'Zoom to original size', enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow,
                           'Ctrl+F', 'fit-window', u'Zoom follows window size',
                           checkable=True, enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', u'Zoom follows window width',
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action('&Edit Label', self.editLabel,
                      'Ctrl+E', 'edit', u'Modify the label of the selected Box',
                      enabled=False)
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action('Shape &Line Color', self.chshapeLineColor,
                                icon='color_line', tip=u'Change the line color for this specific shape',
                                enabled=False)
        shapeFillColor = action('Shape &Fill Color', self.chshapeFillColor,
                                icon='color', tip=u'Change the fill color for this specific shape',
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText('Show/Hide Label Panel')
        labels.setShortcut('Ctrl+Shift+L')

        # Lavel list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = struct(save=save, saveAs=saveAs, open=open, close=close, resetAll = resetAll,
                              lineColor=color1, create=create, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              loadTemplate=loadTemplate,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, color2, upscaleAll, downscaleAll, scaleDown, scaleUp, scaleRight, scaleLeft,
                                        moveDown, moveUp, moveRight, moveLeft, flipHorizontal ),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction("Auto Saving", self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction("Single Class Mode", self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None

        addActions(self.menus.file,
                   (open, opendir, changeSavedir, openAnnotation, self.menus.recentFiles, loadTemplate, save, saveAs, close, resetAll, quit))
        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, None, create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, opendir, changeSavedir, openNextImg, openPrevImg, save, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = settings.get(SETTING_WIN_POSE, QPoint(0, 0))
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, Shape.line_color))
        self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, Shape.fill_color))
        Shape.line_color = self.lineColor
        Shape.fill_color = self.fillColor
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

    ## Support Functions ##

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)
        self.updateLineColors()

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen([self.screencastViewer, self.screencast])

    def showInfoDialog(self):
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self, item=None):
        if not self.canvas.editing():
            return
        item = item if item else self.currentItem()
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            self.setDirty()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    # Add chris
    def btnstate(self, item= None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item: # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count()-1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.close()
            s.append(shape)
            self.addLabel(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            else:
                shape.line_color = generateColorByText(label)

            if fill_color:
                shape.fill_color = QColor(*fill_color)
            else:
                shape.fill_color = generateColorByText(label)
            
            self.addLabel(shape)

        self.canvas.loadShapes(s)
        self.updateLineColors()

    def saveLabels(self, annotationFilePath):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb()
                        if s.line_color != self.lineColor else None,
                        fill_color=s.fill_color.getRgb()
                        if s.fill_color != self.fillColor else None,
                        points=[(p.x(), p.y()) for p in s.points],
                       # add chris
                        difficult = s.difficult)

        shapes = [format_shape(shape) for shape in self.canvas.shapes]
        # Can add differrent annotation formats here
        try:
            if self.usingPascalVocFormat is True:
                print ('Img: ' + self.filePath + ' -> Its xml: ' + annotationFilePath)
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False


    def loadTemplate (self):
        self.labelList.clear()
        scriptPath = os.path.dirname (os.path.realpath(__file__) )
        xmlPath = os.path.join(scriptPath, "template.xml")
        print ("Loading template from ", xmlPath)
        if os.path.isfile(xmlPath):
            self.loadPascalXMLByFilename(xmlPath)

        # we want to resize all shapes to current image size 
        if self.templateImgSize is not None:
            print ("Size of the template image ", str(self.templateImgSize))
            
            # how big is our image?
            print ("Size of loaded image: ", str(self.imgShape))
            
            # first we need the size of the template 
            # we assume here that the orientation is correct already,
            # so we ignore rotated images (which only appear in very few cases in our RNSA data set)
            factorX = float(self.templateImgSize[0])/float(self.imgShape[1])
            factorY = float(self.templateImgSize[1])/float(self.imgShape[0]) # these are swapped, size vs shape etc
            print ("Scaling factor for X: ", str(factorX))
            print ("Scaling factor for Y: ", str(factorY))
            factor = factorY
            if factorX > factorY:
                factor = factorX
            
            for shape in self.canvas.shapes:
                for p in shape.points:
                    p.setX (p.x()/factor)
                    p.setY (p.y()/factor)

            if self.filePath:
                print (self.filePath)
                print (self.defaultSaveDir)
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
                savedFileName = os.path.join (os.path.dirname(self.filePath),  savedFileName)
                print ("Saving XML to " + str(savedFileName ))
                annotationFilePath = savedFileName 
                self._saveFile(annotationFilePath)
                self.setDirty()

        self.labelList.clearSelection()

        return True 
    

    def moveUp (self):
        #print ("Moving UP")
        for shape in self.canvas.shapes:
            for p in shape.points:
                p.setX(p.x()) 
                p.setY(p.y()-25)
        self.setDirty()
        self.paintCanvas()
        return True

    def moveLeft (self):
        #print ("Moving LEFT")
        for shape in self.canvas.shapes:
            for p in shape.points:
                p.setX(p.x()-25) 
                p.setY(p.y())
        self.setDirty()
        self.paintCanvas()
        return True

    def moveRight (self):
        #print ("Moving RIGHT")
        for shape in self.canvas.shapes:
            for p in shape.points:
                p.setX(p.x()+25) 
                p.setY(p.y())
        self.setDirty()
        self.paintCanvas()
        return True



    def flipHorizontal (self):
        #print ("Flipping horizontally")
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        for shape in self.canvas.shapes:
            for p in shape.points:
                p.setX(w2 - p.x()) 
        self.setDirty()
        self.paintCanvas()
        return True



    def moveDown (self):
        #print ("Moving DOWN")
        for shape in self.canvas.shapes:
            for p in shape.points:
                p.setX(p.x()) 
                p.setY(p.y()+25)
        self.setDirty()
        self.paintCanvas()
        return True




    def upscaleX (self, factor, current = True):
        for shape in self.canvas.shapes:
            item = self.currentItem()
            if item is not None:
                if current == True:
                    #print ("Compare" + str(shape.label) + " -- " + item.text())
                    if shape.label != item.text():
                        continue
            minX = 999999999
            maxX = -1
            for p in shape.points:
                minX = min(p.x(), minX)
                maxX = max(p.x(), maxX)
            distX = maxX - minX
            
            for p in shape.points:
                if p.x() == minX:
                    p.setX(minX - distX*factor)
                if p.x() == maxX:
                    p.setX(maxX + distX*factor)
        return True



    def upscaleY (self, factor, current = True):
        for shape in self.canvas.shapes:
            item = self.currentItem()
            if item is not None:
                if current == True:
                    #print ("Compare" + str(shape.label) + " -- " + item.text())
                    if shape.label != item.text():
                        continue
            minY = 999999999
            maxY = -1
            for p in shape.points:
                minY = min(p.y(), minY)
                maxY = max(p.y(), maxY)
            distY = maxY - minY
            
            for p in shape.points:
                if p.y() == minY:
                    p.setY(minY - distY*factor)
                if p.y() == maxY:
                    p.setY(maxY + distY*factor)
        return True


            
    def upscaleAll (self):
        self.upscaleX(0.05)
        self.upscaleY(0.05)
        self.setDirty()
        self.paintCanvas()
        return True


    def downscaleAll(self):
        self.upscaleX(-0.05)
        self.upscaleY(-0.05)
        self.setDirty()
        self.paintCanvas()
        return True



    def scaleUp (self):
        #self.upscaleX(0.05)
        self.upscaleY(0.05)
        self.setDirty()
        self.paintCanvas()
        return True

    def scaleDown (self):
        #self.upscaleX(0.05)
        self.upscaleY(-0.05)
        self.setDirty()
        self.paintCanvas()
        return True

    def scaleRight (self):
        self.upscaleX(0.05)
        #self.upscaleY(0.05)
        self.setDirty()
        self.paintCanvas()
        return True

    def scaleLeft (self):
        self.upscaleX(-0.05)
        #self.upscaleY(0.05)
        self.setDirty()
        self.paintCanvas()
        return True



        

    def colorForKey (self, key):
        r, g, b = 255, 255, 0
        if key[:3] == "MCP":
            r,g,b = 0, 163, 214 # cyan
        if key[:3] == "PIP":
            r,g,b = 212, 0, 179 # violet
        if key[:3] == "Rad":
            r,g,b = 253, 0, 0  # reddish
        if key[:3] == "Uln":
            r,g,b = 236, 179, 20  # goldish
        if key[:3] == "Wri":
            r,g,b = 0, 160, 0  # green
        if key[:3] == "DIP": 
            r,g,b = 255, 120, 0 #orange 
        r,g,b = r/255, g/255, b/255
        scaleFactor = 0
        r = (1-scaleFactor)*r #+ scaleFactor*1
        g = (1-scaleFactor)*g #+ scaleFactor*1
        b = (1-scaleFactor)*b #+ scaleFactor*1
        return r, g, b

            
    def updateLineColors (self):
        for shape in self.canvas.shapes:
            r, g, b = self.colorForKey (shape.label)
            r = int(r*255)
            g = int(g*255)
            b = int(b*255)
            #print (str(r)+","+ str(g)+"," +str(b))
            shape.line_color = QColor( r, g, b)
            shape.vertex_fill_color = QColor( r, g, b)
            shape.fill_color = QColor( r, g, b, 50)
            shape.select_fill_color = QColor( r, g, b, 96)
            shape.hvertex_fill_color = QColor( r, g, b, 96)
        return True
            




    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        print ("Reading from ", filePath)
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        unicodeFilePath = ustr(filePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            index = self.mImgList.index(unicodeFilePath)
            fileWidgetItem = self.fileListWidget.item(index)
            fileWidgetItem.setSelected(True)

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                            
            image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            
            # convrt back and forth 
            rgbimage = cv2.imread(unicodeFilePath, 0)
            rgbimage = pseudoRGB (rgbimage)
            image = array2qimage (rgbimage, normalize = False)


            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.imgShape = rgbimage.shape # actually we could just look at the canvas but its ok this way too.
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            if self.usingPascalVocFormat is True:
                if self.defaultSaveDir is not None:
                    basename = os.path.basename(
                        os.path.splitext(self.filePath)[0]) + XML_EXT
                    xmlPath = os.path.join(self.defaultSaveDir, basename)
                    print ("Loading XML from ", xmlPath)
                    self.loadPascalXMLByFilename(xmlPath)
                else:
                    xmlPath = os.path.splitext(filePath)[0] + XML_EXT
                    if os.path.isfile(xmlPath):
                        print ("Loading XML from ", xmlPath)
                        self.loadPascalXMLByFilename(xmlPath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
                self.labelList.item(self.labelList.count()-1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ""

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ""

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings.save()
    ## User Dialogs ##

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save annotations to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDirDialog(self, _value=False):
        if not self.mayContinue():
            return

        defaultOpenDirPath = '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                     '%s - Open Directory' % __appname__, defaultOpenDirPath,  
                                                     QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        self.importDirImages(dirpath)
        
    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
         if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                self.labelFile.toggleVerify()

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedir()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = os.path.basename(self.filePath)
                savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog())

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            return dlg.selectedFiles()[0]
        return ''

    def _saveFile(self, annotationFilePath):
        if annotationFilePath and self.saveLabels(annotationFilePath):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            # Change the color for all shape lines:
            Shape.line_color = self.lineColor
            self.canvas.update()
            self.setDirty()

    def chooseColor2(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.fillColor = color
            Shape.fill_color = self.fillColor
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.templateImgSize = tVocParseReader.imgSize 
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join(
                         os.path.dirname(sys.argv[0]),
                         'data', 'predefined_classes.txt'))
    win.show()
    return app, win


def main(argv=[]):
    '''construct main app and run it'''
    app, _win = get_main_app(argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
