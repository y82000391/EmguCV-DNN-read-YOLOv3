using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace EmguCV_ObjectDetection_YOLOv3
{
    class Program
    {
        //取出DNN Net的output layer的名稱
        static String[] GetOutputsNames(Net net)
        {
            int[] outLayers = net.UnconnectedOutLayers;
            string[] layerNames = net.LayerNames;

            String[] names = new string[outLayers.Length];
            for (int i = 0; i < outLayers.Length; i++)
            {
                names[i] = layerNames[outLayers[i] - 1];
            }

            return names;
        }

        static void Main(string[] args)
        {
            //讀取DNN Net
            Net Darknet = DnnInvoke.ReadNetFromDarknet(
                          @"model\yolov3-tiny.cfg",
                          @"model\yolov3-tiny.weights");
            //讀取coco dataset object names
            string[] ObjectNames = File.ReadAllLines(@"model\coco.names");

            Darknet.SetPreferableBackend(Emgu.CV.Dnn.Backend.OpenCV);
            Darknet.SetPreferableTarget(Target.Cpu);

            var image = new Image<Bgr, byte>(@"image\dog.jpg");

            Mat inputBlob = DnnInvoke.BlobFromImage(image, 1.0 / 255.0, new Size(416, 416), new MCvScalar(0), true, false);

            VectorOfMat output = new VectorOfMat();
            Darknet.SetInput(inputBlob);
            Darknet.Forward(output, GetOutputsNames(Darknet));

            //新增三個List，包含物件的Rectangle, 物件分數, 物件的index
            List<Rectangle> rects = new List<Rectangle>();
            List<float> scores = new List<float>();
            List<int> objIndexs = new List<int>();
            
            //取出YOLOv3執行output layer，共會有三層
            for (int l = 0; l < output.Size; l++)
            {
                var boxes = output[l];
                int resultRows = boxes.SizeOfDimension[0];
                int resultCols = boxes.SizeOfDimension[1];

                float[] temp = new float[resultRows * resultCols];
                Marshal.Copy(boxes.DataPointer, temp, 0, temp.Length);

                for (int i = 0; i < resultRows; i++)
                {
                    //取出sub array, 從第六個位置開始抓，以此例會抓到80筆資料(對應到coco 80個物件)
                    var subMat = new Mat(boxes.Row(i), new Rectangle(5, 0, resultCols - 5, 1));
                    //找出這80筆資料分數最高者
                    subMat.MinMax(out double[] minValues, out double[] maxValues, out Point[] minPoints, out Point[] maxPoints);

                    //若是判斷分數大於0，則進行下一步
                    if (maxValues[0] > 0)
                    {
                        //取出該物件的rectangle
                        int centerX = (int)(temp[i * resultCols + 0] * image.Width);
                        int centerY = (int)(temp[i * resultCols + 1] * image.Height);
                        int width = (int)(temp[i * resultCols + 2] * image.Width);
                        int height = (int)(temp[i * resultCols + 3] * image.Height);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;
                        Rectangle rect = new Rectangle(left, top, width, height);

                        //將rectangle, score, object index加入List
                        rects.Add(rect);
                        scores.Add((float)maxValues[0]);
                        objIndexs.Add(maxPoints[0].X);
                    }
                }
            }

            var resultImage = image.Clone();
            //把偵測出的物件繪圖
            for (int i = 0; i < rects.Count; i++)
            {
                resultImage.Draw(rects[i], new Bgr(100, 100, 255), 2);
                var objName = ObjectNames[objIndexs[i]];
                resultImage.Draw(objName, new Point(rects[i].X, rects[i].Y - 10), FontFace.HersheyTriplex, 0.5, new Bgr(255, 100, 100));
                resultImage.Draw(scores[i].ToString(), new Point(rects[i].X, rects[i].Y + 10), FontFace.HersheyTriplex, 0.5, new Bgr(100, 255, 100));
            }

            resultImage.Save(@"image\result.jpg");

            //利用NMS把重複位置的rectangle去除
            var selectedObj = DnnInvoke.NMSBoxes(rects.ToArray(), scores.ToArray(), 0.2f, 0.3f);

            //再把偵測出的物件繪圖
            for (int i = 0; i < rects.Count; i++)
            {
                //只畫出被保留下來的rectangle
                if (selectedObj.Contains(i))
                {
                    image.Draw(rects[i], new Bgr(100, 100, 255), 2);
                    var objName = ObjectNames[objIndexs[i]];
                    image.Draw(objName, new Point(rects[i].X, rects[i].Y - 10), FontFace.HersheyTriplex, 0.5, new Bgr(255, 100, 100));
                    image.Draw(scores[i].ToString(), new Point(rects[i].X, rects[i].Y + 10), FontFace.HersheyTriplex, 0.5, new Bgr(100, 255, 100));
                }
            }

            image.Save(@"image\NMSresult.jpg");

        }
    }
}
