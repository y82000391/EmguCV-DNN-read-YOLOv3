# EmguCV-DNN-read-YOLOv3
using YOLOv3 with EmguCV

程式說明網誌版請參考：https://yy-programer.blogspot.com/2020/07/emgucv-dnn-yolov3.html

<img src="https://1.bp.blogspot.com/-OZ2veEt6nYA/XwQeiYoXKtI/AAAAAAAAY8g/JfXdbUCPsjo_bDmhup4Nl8e1sY2jirSEACNcBGAsYHQ/w781-h430/013.jpg"/>

沒想到OpenCV 3.4.2版以上已經開始支援YOLOv3

而目前EmguCV也已經更新到4.3.0版本

以C#開發者來說，這比編譯darknet wrapper還來的更方便啊!!

就來看看怎麼在EmguCV中使用YOLOv3吧!!

新增一個C# 的console專案 (.net framework 4.6)

於參考上設定管理Nuget套件

在瀏覽的頁面上，搜尋emgu.cv

搜尋時，那個"."非常重要，如果是emgucv，會搜尋到別的套件

Emgu.CV新版在nuget上包裝的非常方便

把開發套件與執行所需的元件切開

如下圖：

<img src="https://1.bp.blogspot.com/-uqsBPXMbPds/XwPkwa7oXCI/AAAAAAAAY5o/lQo1yfmFp54aaBMc6Cl6cgHbYbedXIwcwCNcBGAsYHQ/w781-h664/001.png"/>

開發時需安裝Emgu.CV這個套件，若需要執行則依據你要執行的環境

若是CPU執行環境，則安裝Emgu.CV.runtime.windows

若是在有GPU的執行環境，則安裝Emgu.CV.runtime.windows.cuda

這裡如果你在windows的CPU環境要開發與測試

則安裝

Emgu.CV 與 Emgu.CV.runtime.windows

在專案的參考上會多出如下圖項目

<img src="https://1.bp.blogspot.com/-CTmwoHI6Odg/XwPk4qyEY5I/AAAAAAAAY5s/ccE8zih5anU4EW65h32R13mMxug0oqyzQCNcBGAsYHQ/s320/003.png"/>

接著在專案屬性頁面，在建置中把平台目標設定為x64

<img src="https://1.bp.blogspot.com/-d1jRiz3kWBo/XwPlC749xaI/AAAAAAAAY5w/eyXuDlQkG-gGZAZNXWVzxNt4QJMhXtZPgCNcBGAsYHQ/w625-h358/002.png"/>

接下來要讀取YOLOv3的檔案

需下載

YOLOv3所需的cfg、weights、names檔案

YOLOv3官網：https://pjreddie.com/darknet/yolo/

這裡我們測試下載YOLOv3-Tiny，執行速度比較快，適合在CPU上運行

下載位置如下圖：

<img src="https://1.bp.blogspot.com/-4FarfSjfU98/XwPl7t_g24I/AAAAAAAAY6I/NLclajnhncQ-BLR6FOF6c5VBdpqkUe28gCNcBGAsYHQ/w625-h576/0041.png"/>

由於YOLOv3都是用COCO Dataset進行訓練

因此還要下載COCO Dataset的coco.names

可以到YOLOv3原始Github的空間下載：https://github.com/pjreddie/darknet/tree/master/data

如下圖：

<img src="https://1.bp.blogspot.com/-h5pKhj0wyyU/XwPlOQ3bXdI/AAAAAAAAY54/BSY2QtuH_6U9WrHMOitmQ-gFAJIMZiKHQCNcBGAsYHQ/w781-h525/004.png"/>

下載完後，在程式的執行目錄(bin\debug)新增一個model的資料夾

把剛剛下載的三個檔案放到該資料夾

下載一張常用的影像偵測測試影像dog.jpg

並放到執行目錄新增image的資料夾中(bin\debug\image)

<img src="https://1.bp.blogspot.com/-4Xlknixnw1I/XwPmDZz_cmI/AAAAAAAAY6M/NwMCyh9SKZIKlvO4EYfhBCeOA4kgbgJ8QCNcBGAsYHQ/w500-h375/dog.jpg"/>

讀取YOLOv3的檔案與影像的程式如下：

<img src="https://1.bp.blogspot.com/-ugDHX6aR9e4/XwPmJBrU4TI/AAAAAAAAY6Q/ewX9fzdq_jgFJvWSIon6ZNoaufsD5U5HwCNcBGAsYHQ/w781-h299/005.png"/>

ReadNetFromXXX 可以讀取很多不同來源的Model

OpenCV真的是融合了非常多的類神經網路，ReadNetFromDarknet就是讀取YOLO系列

這邊需要注意DNN讀取影像的方式

參數的設定請參考上圖的BlobFromImage方法

大家一定好奇為何size是416x416 ??

這是唯一會需要調整的部份，可依據YOLO cfg檔裡面的設定做調整，如下圖：

<img src="https://1.bp.blogspot.com/-7AjKzUTUg3Y/XwQNZMH0vDI/AAAAAAAAY78/kFyExL9-tiIczENGZ9HnJlm2bUGcDILcACNcBGAsYHQ/w371-h379/011.jpg"/>

(YOLO cfg檔，預設都是416 x 416)


接著在執行Net.Forward時，需要去抓取該Net的output layer的名稱集合

透過下列的方法取得，如下圖：

<img src="https://1.bp.blogspot.com/-ylrpNA2rFVs/XwPmPpT-DSI/AAAAAAAAY6Y/8Q3gB6KbA5kTqqexMPCpjjULBMHYCD15wCNcBGAsYHQ/w500-h320/006.png"/>

執行的結果儲存在output中

YOLOv3的output layer有三層，分別是yolo_82、yolo_94、yolo_106

因此output會包含三個2維陣列

透過for迴圈取出每一層的結果

這裡解釋一下output的資料結構

每一層的output都是一個2維陣列，代表意義如下圖

<img src="https://1.bp.blogspot.com/-IEc6J33TyHg/XwQQ5gTYDJI/AAAAAAAAY8U/XRea06SvXb8hS6MoiiPCmx0nEtySls87wCNcBGAsYHQ/w625-h251/009.jpg"/>

每一列都是一個偵測目標的資訊

前4個欄位是rectangle資訊，第6個欄位開始，是該目標對應到coco dataset的辨識分數

因此要找到該目標到底是被偵測成什麼物件

就是從第6個欄位開始到第85個欄位(共80個欄位)，比較看哪一個欄位的信任分數最高

假設找到第18個欄位數值最高

就代表該目標被YOLO辨識為coco dataset中的第18個物件

而這第18個物件是什麼呢? 請參照coco.names，找到第18個就知道了，如下圖：

<img src="https://1.bp.blogspot.com/-SaRic2tGM3Q/XwQPDpUYPKI/AAAAAAAAY8I/HM548vWCEQATPhWUF9H9z1hxMlSevkEnwCNcBGAsYHQ/s0/012.jpg"/>

接著在程式中新增三個所需的List物件，分別儲存偵測物件的矩形資訊、辨識分數、物件ID

利用Mat的MinMax方法取出第6~85欄位的最大值與最大值的位置

若判斷分數大於0，才進行後續擷取rectangle資訊

把三層output layer用for迴圈跑完後

剛剛新增的rects、scores、objIndexs 已經塞好了YOLOv3-tiny 偵測出所有有效目標

<img src="https://1.bp.blogspot.com/-TeUNAN3tpak/XwPm7_Uay8I/AAAAAAAAY60/N5bsAPNmABYENuztFdgSUJMD5J22ldJdwCNcBGAsYHQ/w976-h889/007.png"/>

接著透過迴圈與image.Draw的方法，把相關資訊畫在影像上

呈現的結果如下圖

<img src="https://1.bp.blogspot.com/-PG7x1rfskeA/XwPnD3f2ilI/AAAAAAAAY64/1h3FazwFKbEXkX980U_eGFg_w7FT79uewCNcBGAsYHQ/w500-h375/result.jpg"/>

可以看到狗、腳踏車、汽車等物件都被偵測出來

但在汽車的區域被重複偵測出一個卡車的物件

OpenCV內建一個可以去除重複物件的方法 NMSBoxes

使用方式如下圖

<img src="https://1.bp.blogspot.com/-p0YJ4B00L_Q/XwPnYewZ_MI/AAAAAAAAY7E/Ii7TI4fN66sE4zWbGEoVrVYN7xKOUqWjgCNcBGAsYHQ/w976-h314/008.png"/>

NMSBoxes帶入rectangle與score的陣列，以及一些閥值參數

回傳的是保留下來的陣列序號(index)

接著把去除重複的物件匯出，如下圖

<img src="https://1.bp.blogspot.com/-zDh_2tG23ic/XwPnlrtK7bI/AAAAAAAAY7M/lDfJfGotjlEP87o2Y4_Dvi7DNQKCHItKQCNcBGAsYHQ/w500-h375/NMSresult.jpg"/>

原本被偵測卡車的區域就被移除了，只剩下汽車

這在畫面中有複雜場景時相當好用

如下幾個對照可參考

<img src="https://1.bp.blogspot.com/-qPCAcuyaBM8/XwP2k7ylxbI/AAAAAAAAY7s/ITa39aPgObw3y7eQ85yJAvrOzIJdV-oQQCNcBGAsYHQ/w625-h351/result.jpg"/>
<img src="https://1.bp.blogspot.com/-5k3LlnTKvQI/XwP2k8IlyII/AAAAAAAAY7o/uhIopT7eHssoekUmP-WFCtst2XCMYMkRwCNcBGAsYHQ/w625-h351/NMSresult.jpg"/>

以上就是利用EmguCV使用YOLOv3進行物件偵測

大家可以試著換成YOLOv3來跑看看，會比tiny的效果好很多

由於很多人來問程式碼的部分

之後都會將每篇的source code上傳至github上，方便大家取用參考嘍!!


