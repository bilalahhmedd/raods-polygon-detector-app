
from model import *
from utils import *

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic

from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter
import matplotlib.pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(959, 744)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 160, 681))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btn_openDir = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_openDir.setFont(font)
        self.btn_openDir.setMouseTracking(True)
        self.btn_openDir.setAutoFillBackground(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./ui/Open in Browser_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_openDir.setIcon(icon)
        self.btn_openDir.setFlat(True)
        self.btn_openDir.setObjectName("btn_openDir")
        self.verticalLayout.addWidget(self.btn_openDir)
        self.btn_nextImage = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_nextImage.setFont(font)
        self.btn_nextImage.setMouseTracking(True)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("./ui/Circled Right 2_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_nextImage.setIcon(icon1)
        self.btn_nextImage.setDefault(False)
        self.btn_nextImage.setFlat(True)
        self.btn_nextImage.setObjectName("btn_nextImage")
        self.verticalLayout.addWidget(self.btn_nextImage)
        self.btn_prevImage = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_prevImage.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("./ui/Circled Left 2_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_prevImage.setIcon(icon2)
        self.btn_prevImage.setFlat(True)
        self.btn_prevImage.setObjectName("btn_prevImage")
        self.verticalLayout.addWidget(self.btn_prevImage)
        self.btn_zoom_in = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_zoom_in.setFont(font)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("./ui/Collapse_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_zoom_in.setIcon(icon3)
        self.btn_zoom_in.setFlat(True)
        self.btn_zoom_in.setObjectName("btn_zoom_in")
        self.verticalLayout.addWidget(self.btn_zoom_in)
        self.btn_zoom_out = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_zoom_out.setFont(font)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("./ui/Expand_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_zoom_out.setIcon(icon4)
        self.btn_zoom_out.setFlat(True)
        self.btn_zoom_out.setObjectName("btn_zoom_out")
        self.verticalLayout.addWidget(self.btn_zoom_out)
        self.btn_process = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_process.setFont(font)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("./ui/Automatic_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_process.setIcon(icon5)
        self.btn_process.setFlat(True)
        self.btn_process.setObjectName("btn_process")
        self.verticalLayout.addWidget(self.btn_process)


        self.btn_wrap = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_wrap.setFont(font)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("./ui/Align Justify_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_wrap.setIcon(icon6)
        self.btn_wrap.setFlat(True)
        self.btn_wrap.setObjectName("btn_wrap")
        self.verticalLayout.addWidget(self.btn_wrap)
        self.btn_save = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(10)
        self.btn_save.setFont(font)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("./ui/Circled Down_48px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_save.setIcon(icon7)
        self.btn_save.setFlat(True)
        self.btn_save.setObjectName("btn_save")
        self.verticalLayout.addWidget(self.btn_save)
        self.txt_xml = QtWidgets.QTextEdit(self.centralwidget)
        self.txt_xml.setGeometry(QtCore.QRect(690, 10, 261, 271))
        self.txt_xml.setObjectName("txt_xml")
        self.lbl_wrapImage = QtWidgets.QLabel(self.centralwidget)
        self.lbl_wrapImage.setGeometry(QtCore.QRect(170, 420, 511, 261))
        self.lbl_wrapImage.setText("")
        self.lbl_wrapImage.setPixmap(QtGui.QPixmap(""))
        self.lbl_wrapImage.setScaledContents(True)
        self.lbl_wrapImage.setObjectName("lbl_wrapImage")
        self.lbl_fileList = QtWidgets.QLabel(self.centralwidget)
        self.lbl_fileList.setGeometry(QtCore.QRect(780, 330, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(12)
        self.lbl_fileList.setFont(font)
        self.lbl_fileList.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lbl_fileList.setScaledContents(True)
        self.lbl_fileList.setObjectName("lbl_fileList")
        self.lbl_rawImage = QtWidgets.QLabel(self.centralwidget)
        self.lbl_rawImage.setGeometry(QtCore.QRect(170, 10, 511, 401))
        self.lbl_rawImage.setText("")
        self.lbl_rawImage.setPixmap(QtGui.QPixmap(""))
        self.lbl_rawImage.setScaledContents(True)
        self.lbl_rawImage.setObjectName("lbl_rawImage")
        self.fileList_widget = QtWidgets.QListWidget(self.centralwidget)
        self.fileList_widget.setGeometry(QtCore.QRect(690, 360, 261, 321))
        self.fileList_widget.setObjectName("fileList_widget")
        self.btn_lane = QtWidgets.QCheckBox(self.centralwidget)
        self.btn_lane.setGeometry(QtCore.QRect(850, 290, 91, 17))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_lane.setFont(font)
        self.btn_lane.setObjectName("btn_lane")
        self.btn_polygon = QtWidgets.QCheckBox(self.centralwidget)
        self.btn_polygon.setGeometry(QtCore.QRect(690, 290, 111, 17))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_polygon.setFont(font)
        self.btn_polygon.setObjectName("btn_polygon")
        
        self.btn_toggle_polygon_xml = QtWidgets.QRadioButton(self.centralwidget)
        self.btn_toggle_polygon_xml.setGeometry(QtCore.QRect(690, 310, 101, 17))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_toggle_polygon_xml.setFont(font)
        self.btn_toggle_polygon_xml.setObjectName("btn_toggle_polygon_xml")
        self.btn_toggle_lane_xml = QtWidgets.QRadioButton(self.centralwidget)
        self.btn_toggle_lane_xml.setGeometry(QtCore.QRect(850, 310, 82, 17))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_toggle_lane_xml.setFont(font)
        self.btn_toggle_lane_xml.setObjectName("btn_toggle_lane_xml")
        
        
        self.model_loading_bar = QtWidgets.QProgressBar(self.centralwidget)
        self.model_loading_bar.setGeometry(QtCore.QRect(0, 690, 951, 16))
        self.model_loading_bar.setProperty("value", 0)
        self.model_loading_bar.setObjectName("model_loading_bar")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 959, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        
        
        # #
        self.btn_polygon.setEnabled(False)
        self.btn_lane.setEnabled(False)
        self.btn_toggle_polygon_xml.setEnabled(False)
        self.btn_toggle_lane_xml.setEnabled(False)

        # #

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_openDir.setText(_translate("MainWindow", "Open Dir"))
        self.btn_nextImage.setText(_translate("MainWindow", "Next Image"))
        self.btn_prevImage.setText(_translate("MainWindow", "Prev Image"))
        self.btn_zoom_in.setText(_translate("MainWindow", "Zoom In"))
        self.btn_zoom_out.setText(_translate("MainWindow", "Zoom Out"))
        self.btn_process.setText(_translate("MainWindow", "Run"))
        self.btn_wrap.setText(_translate("MainWindow", "Wrap"))
        self.btn_save.setText(_translate("MainWindow", "Save"))
        self.lbl_fileList.setText(_translate("MainWindow", "Files List"))
        self.btn_polygon.setText(_translate("MainWindow", "Show Polygon"))
        self.btn_lane.setText(_translate("MainWindow", "Show Lane"))
        
        self.btn_toggle_polygon_xml.setText(_translate("MainWindow", "Polygon XML"))
        self.btn_toggle_lane_xml.setText(_translate("MainWindow", "Lane XML"))
        
    


# Derived from the Ui_MainWindow to avoid exceptions.
class Main(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)
        self.__connectEvents()
        
        self.wrapImage_clicked = False
        self.qimage = QImage()
        self.qpixmap = QPixmap()
        
        self.qimage_scaled = QImage()
        self.zoomX = 1              # zoom factor w.r.t size of qlabel_image
        self.position = [0, 0]      # position of top left corner of lbl_wrap_image
        self.is_processed_btn_clicked = False # To check if the process button is clicked, used to show wrap and polygon image

        self.images = None
        
    def __connectEvents(self):
        self.btn_openDir.clicked.connect(self.selectDir)
        self.btn_nextImage.clicked.connect(self.display_next_image)
        self.btn_prevImage.clicked.connect(self.display_prev_image)
        self.fileList_widget.itemClicked.connect(self.fileList_item_click)
        self.btn_polygon.clicked.connect(self.set_images_in_QPix)
        self.btn_lane.clicked.connect(self.set_images_in_QPix)
        self.btn_process.clicked.connect(self.process_image)
        self.btn_wrap.clicked.connect(self.set_wrapImage)
        
        self.btn_toggle_polygon_xml.clicked.connect(self.display_relative_xml)
        self.btn_toggle_lane_xml.clicked.connect(self.display_relative_xml)
        self.btn_zoom_in.clicked.connect(self.zoomPlus)
        self.btn_zoom_out.clicked.connect(self.zoomMinus)
        
        self.btn_save.clicked.connect(self.saveImage)
        
    def check_img_state(self):
        state_poly_lane = False
        state_poly = False
        state_lane = False
        
        exception_state = None
        if (self.btn_lane.isChecked() and self.btn_polygon.isChecked()):
            state_poly_lane = True
        elif (self.btn_lane.isChecked()):
            state_lane = True
        elif self.btn_polygon.isChecked():
            state_poly = True
        else:
            pass
        
        return state_poly_lane,state_poly,state_lane, exception_state
            
        
    def set_images_in_QPix(self):
        state_poly_lane,state_poly,state_lane, _ = self.check_img_state()
        exception_state = True
        self.lbl_rawImage.setPixmap(QtGui.QPixmap(None))
        
        if state_poly_lane:
            if self.c_polygon_lane_PixMap:
                self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.c_polygon_lane_PixMap))
                exception_state = False
        
        if state_lane:
            if self.c_lane_pixMap:
                self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.c_lane_pixMap))
                exception_state = False
        
        if state_poly:
            if self.c_polygon_pixMap:
                self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.c_polygon_pixMap))
                exception_state = False
        
        if exception_state:
            print("Selected image not found!")
            self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.c_original_pixMap))
            
            
    def selectDir(self):
        ''' Select a directory, make list of images in it and display the first image in the list. '''
        # open 'select folder' dialog box
    
        self.folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))  
        
        if not self.folder:
            QtWidgets.QMessageBox.warning(self, 'No Folder Selected', 'Please select a valid Folder')
            return
        
        self.im_index = 0
        self.images = getImages(self.folder)
        self.items = [QtWidgets.QListWidgetItem(img['name']) for img in self.images]
        
        self.fileList_widget.clear()
        for item in self.items:
            self.fileList_widget.addItem(item)
        
        
        if len(self.images) > 0:
            self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.images[self.im_index]['path']))
    
    def fileList_item_click(self, item):
        self.clearProcessedCache()
        self.resetZoom()
        self.lbl_wrapImage.setPixmap(QtGui.QPixmap(""))
        self.im_index = self.items.index(item)
        self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.images[self.im_index]['path']))
        self.txt_xml.setPlainText("")
        self.is_processed_btn_clicked = False
        self.wrapImage_clicked = False

            
    def display_next_image(self):
        try:
            if self.im_index < len(self.images) -1:
                self.im_index += 1
                self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.images[self.im_index]['path']))
                self.lbl_wrapImage.setPixmap(QtGui.QPixmap(None))
                self.is_processed_btn_clicked = False
                self.txt_xml.setPlainText("")
                self.wrapImage_clicked = False
                self.clearProcessedCache()
            else:
                QtWidgets.QMessageBox.warning(self, 'Sorry', 'No more Images!')
        except AttributeError:
            pass

            
    def display_prev_image(self):
        try:
            if self.im_index:
                if self.im_index > 0:
                    self.im_index -= 1
                    self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.images[self.im_index]['path']))
                    self.lbl_wrapImage.setPixmap(QtGui.QPixmap(None))
                    self.is_processed_btn_clicked = False
                    self.txt_xml.setPlainText("")
                    self.wrapImage_clicked = False
                    self.clearProcessedCache()
                else:
                    QtWidgets.QMessageBox.warning(self, "Sorry", 'No more Images!')
        except AttributeError:
            pass
    
    # so here we are warping image and showing in window    
    def set_wrapImage(self):
        print('warping button pressed')
        
        if self.is_processed_btn_clicked and self.c_lane_wrap_PixMap:
            #self.lbl_wrapImage.setPixmap(QtGui.QPixmap(None))
            self.lbl_wrapImage.setPixmap(QtGui.QPixmap(self.c_lane_wrap_PixMap))

            
            self.qimage = QImage(self.c_lane_wrap_PixMap)
            self.qpixmap = QPixmap(self.lbl_wrapImage.size())
            self.qimage_scaled = self.qimage.scaled(self.lbl_wrapImage.width(),
                                                    self.lbl_wrapImage.height(),
                                                    QtCore.Qt.KeepAspectRatio)
            #self.update()
            
            self.wrapImage_clicked = True
        
            
        
    def update_xml_textBox(self,fname,identifier):
        dir_name = "saved_xmls"
        fname = name_modify_img(fname, identifier=identifier,prefered_ext= "xml")
        tree= ET.parse('./'+dir_name+'/'+fname)
        tree = tree.getroot()
        t = tostring(tree)

        xmlstr = minidom.parseString(ET.tostring(tree)).toprettyxml(indent="   ")
        self.txt_xml.setPlainText(str(xmlstr))
        return xmlstr
    
    def getQPixMap_Image(self, image):
        if type(image) is np.ndarray:
            image = Image.fromarray(np.uint8(image)).convert('RGB')
        
        self.q_image  = ImQT(image) 

        pixmap = QtGui.QPixmap.fromImage(self.q_image)
        return pixmap
    
    def clearProcessedCache(self):
        # Clear pixMaps
        self.c_original_pixMap = None
        self.c_wrap_polygon_PixMap = None
        self.c_polygon_pixMap = None
        self.c_lane_pixMap = None

        # Change these
        self.btn_polygon.setEnabled(False)
        self.btn_lane.setEnabled(False)
        self.is_processed_btn_clicked = False
        self.btn_lane.setChecked(False) 
        self.btn_polygon.setChecked(False) 
        self.btn_toggle_lane_xml.setChecked(False)
        self.btn_toggle_lane_xml.setEnabled(False)
        self.btn_toggle_polygon_xml.setChecked(False)
        self.btn_toggle_polygon_xml.setEnabled(False)
        self.model_loading_bar.setValue(0)
        self.txt_xml.setPlainText("")
        
    def display_relative_xml(self):
        if self.btn_toggle_polygon_xml.isChecked():
            self.update_xml_textBox(self.images[self.im_index]['name'],"-ply")
        if self.btn_toggle_lane_xml.isChecked():
            self.update_xml_textBox(self.images[self.im_index]['name'],"-lane")
            
    
    def handleTimer(self):
        value = self.model_loading_bar.value()
        if value < 100:
            value +=1
            self.model_loading_bar.setValue(value)
        else:
            self.timer.stop()
        
    def process_image(self):
        # Exception needs to be handled here if no images are loaded.
        if self.images == None or (len(self.images) == 0):
            QtWidgets.QMessageBox.warning(self, "Sorry", 'Images are not loaded!')
        else:
            self.c_lane_pixMap = None
            self.c_original_pixMap = None
            self.c_polygon_pixMap = None
            self.c_wrap_polygon_PixMap  = None
            self.c_lane_wrap_PixMap = None
            
            # Timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.handleTimer)
            self.timer.start(10)
            
            self.is_processed_btn_clicked = True
            exist = False

            if not exist and (len(self.images) > 0):
                
                print("Model is Running...")
                try:
                    out,images = evaluate_image(self.images[self.im_index]['path'])
                    original_pixMap = self.getQPixMap_Image(images)
                    self.c_original_pixMap = original_pixMap.copy()
                    del original_pixMap


                    contours,hierarchy = compute_contours(out)
                    polygone, polygon_img, img, polygon_fname = draw_polygon(contours, images)

                    polygon_PixMap = self.getQPixMap_Image(polygon_img)
                    self.c_polygon_pixMap = polygon_PixMap.copy()
                    del polygon_PixMap

                    S1, S2, S3, S4 = get_rect_points(polygone)
                    create_save_xml_data(self.images[self.im_index]['name'],"-ply",S1,S2,S3,S4)
                    self.update_xml_textBox(self.images[self.im_index]['name'],"-ply")

                    self.btn_toggle_polygon_xml.setChecked(True)

                except Exception as e:
                    print(e)

                out_lane,lane_image = evaluate_image_lane(self.images[self.im_index]['path'])
                left,right,contours_lane = compute_contours_lane(out_lane)
                if left != None:
                    lane_img_data,lane_imaged,fname = draw_lane(contours_lane, lane_image,left,right)

                    lane_PixMap = self.getQPixMap_Image(lane_img_data)
                    self.c_lane_pixMap = lane_PixMap.copy()
                    del lane_PixMap

                    polygon_lane_img = cv2.addWeighted((polygon_img),0.5,(lane_img_data),0.5,0.0)
                    polygon_lane_img = self.getQPixMap_Image(polygon_lane_img)
                    self.c_polygon_lane_PixMap = polygon_lane_img.copy()
                    del polygon_lane_img


                    ls1,ls2,ls3,ls4 = get_rect_points_lane(left,right,contours_lane)
                    create_save_xml_data(self.images[self.im_index]['name'],"-lane",ls1,ls2,ls3,ls4) # Saving points

                    # Wrap-Lane Image processing block starts here
                    image=lane_image.resize((1600, 1200), Image.ANTIALIAS)
                    fx=1
                    fy = 1
                    img = image.convert('RGB')
                    img = np.array(img)
                    points = (ls1,ls2,ls3,ls4)
                    pts = points
                    print('points: ',points)
                    wrap_image_lane= four_point_transform(img,pts)
                    print(wrap_image_lane.shape)
                    # correct so far
                    wrap_PixMap= self.getQPixMap_Image(wrap_image_lane.copy())
                    self.c_lane_wrap_PixMap  = wrap_PixMap.copy()
                    #del wrap_PixMap
                if self.c_polygon_pixMap != None:
                    self.btn_polygon.setEnabled(True)
                    self.btn_toggle_polygon_xml.setEnabled(True)
                if self.c_lane_pixMap != None:
                    self.btn_lane.setEnabled(True)
                    self.btn_toggle_lane_xml.setEnabled(True)
                print("Model Finished Successfully!")
            else:
                pass # Can be used if require to work on some already saved image
    def update(self):
        ''' This function actually draws the scaled image to the qlabel_image.
            It will be repeatedly called when zooming or panning.
            So, I tried to include only the necessary operations required just for these tasks. 
        '''
        if not self.qimage_scaled.isNull():
            # check if position is within limits to prevent unbounded panning.
            px, py = self.position
            px = px if (px <= self.qimage_scaled.width() - self.lbl_wrapImage.width()) else (self.qimage_scaled.width() - self.lbl_wrapImage.width())
            py = py if (py <= self.qimage_scaled.height() - self.lbl_wrapImage.height()) else (self.qimage_scaled.height() - self.lbl_wrapImage.height())
            px = px if (px >= 0) else 0
            py = py if (py >= 0) else 0
            self.position = (px, py)

            if self.zoomX == 1:
                self.qpixmap.fill(QtCore.Qt.white)

            # the act of painting the qpixamp
            painter = QPainter()
            painter.begin(self.qpixmap)
            painter.drawImage(QtCore.QPoint(0, 0), self.qimage_scaled,
                    QtCore.QRect(self.position[0], self.position[1], self.lbl_wrapImage.width(), self.lbl_wrapImage.height()) )
            painter.end()
            self.lbl_wrapImage.setPixmap(self.qpixmap)
        else:
            pass
        
    def zoomPlus(self):
        if self.wrapImage_clicked:
            self.zoomX += 1
            px, py = self.position
            px += self.lbl_wrapImage.width()/2
            py += self.lbl_wrapImage.height()/2
            self.position = (px, py)
            self.qimage_scaled = self.qimage.scaled(self.lbl_wrapImage.width() * self.zoomX, self.lbl_wrapImage.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
            self.update()

    def zoomMinus(self):
        if self.wrapImage_clicked:
            if self.zoomX > 1:
                self.zoomX -= 1
                px, py = self.position
                px -= self.lbl_wrapImage.width()/2
                py -= self.lbl_wrapImage.height()/2
                self.position = (px, py)
                self.qimage_scaled = self.qimage.scaled(self.lbl_wrapImage.width() * self.zoomX, self.lbl_wrapImage.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
                self.update()
                
    def resetZoom(self):
        self.zoomX = 1
        self.position = [0, 0]
        self.qimage_scaled = self.qimage.scaled(self.lbl_wrapImage.width() * self.zoomX, self.lbl_wrapImage.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()
        
    def saveImage(self):
        if self.is_processed_btn_clicked:
            state_poly_lane,state_poly,state_lane, _ = self.check_img_state()
            
            if state_poly_lane:
                if self.c_polygon_lane_PixMap:
                    save_image(self.c_polygon_lane_PixMap,self.images[self.im_index]['name'],identifier="-ploylane")
                    exception_state = False
        
            if state_lane:
                if self.c_lane_pixMap:
                    save_image(self.c_lane_pixMap,self.images[self.im_index]['name'],identifier="-lane")
                    exception_state = False

            if state_poly:
                if self.c_polygon_pixMap:
                    save_image(self.c_polygon_pixMap,self.images[self.im_index]['name'],identifier="-poly")
                    self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.c_polygon_pixMap))
                    exception_state = False
            
            if self.wrapImage_clicked:
                save_image(self.c_wrap_polygon_PixMap,self.images[self.im_index]['name'],identifier="-wrap")
                exception_state = False
                
            if exception_state:
                print("Selected image not found!")
                self.lbl_rawImage.setPixmap(QtGui.QPixmap(self.c_original_pixMap))
                
            print("Saved Succesfully")
        else:
            QtWidgets.QMessageBox.warning(self, "Sorry", 'Images are not loaded!')
         


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec())
