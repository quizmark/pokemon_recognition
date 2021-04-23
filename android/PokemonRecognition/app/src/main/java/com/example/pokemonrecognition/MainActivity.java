package com.example.pokemonrecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.os.Handler;
import android.os.PowerManager;
import android.os.StrictMode;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Base64;
import android.util.JsonReader;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.graphics.Bitmap.CompressFormat;

import org.apache.http.NameValuePair;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.params.BasicHttpParams;
import org.apache.http.params.HttpConnectionParams;
import org.apache.http.params.HttpParams;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.NetworkInterface;
import java.net.StandardSocketOptions;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;


import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Locale;
import java.util.Vector;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.DefaultHttpClient;

import android.os.AsyncTask;
import android.util.Log;
import android.widget.Toast;

import de.hdodenhof.circleimageview.CircleImageView;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {
    Camera camera;
    Camera.PreviewCallback frame;
    SurfaceView surfaceView;
    SurfaceHolder surfaceHolder;
    Bitmap bitmapCam;
    TextView textViewName;
    ImageView imageView;



    public String img2str(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        return Base64.encodeToString(byteArray, Base64.DEFAULT);
    }
    public Bitmap rotateBitmap(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }
    public Bitmap cropBitmap(Bitmap bitmap, Rect rect) {
        Bitmap result = Bitmap.createBitmap(rect.right - rect.left, rect.bottom - rect.top, Bitmap.Config.ARGB_8888);
        new Canvas(result).drawBitmap(bitmapCam, -rect.left, -rect.top, null);
        return result;
    }
    public Bitmap resizeBitmap(Bitmap bitmap, int width, int height) {
        Bitmap result = Bitmap.createScaledBitmap(
                bitmap, width, height, false);
        return result;
    }
    public Bitmap getAvt(Bitmap bitmapCam) {
        Rect rectAvt = new Rect();
        rectAvt.bottom = bitmapCam.getHeight() - (bitmapCam.getHeight() - bitmapCam.getWidth()) / 2;
        rectAvt.top = (bitmapCam.getHeight() - bitmapCam.getWidth()) / 2;
        rectAvt.left = 0;
        rectAvt.right = bitmapCam.getWidth();
        Bitmap bmavt = cropBitmap(bitmapCam, rectAvt);
        //bmavt = resizeBitmap(bmavt, 140, 140);
        return bmavt;
    }
    public Bitmap str2img(String avatar){
        byte[] b = Base64.decode(avatar, 1);
        BitmapFactory.Options options = new BitmapFactory.Options();
        Bitmap mBitmap = BitmapFactory.decodeByteArray(b, 0, b.length, options);
        return mBitmap;
    }
    public void checkPermissionCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
        }
    }
    public void checkStrictMode(){
        if (android.os.Build.VERSION.SDK_INT > 9) {
            StrictMode.ThreadPolicy policy =
                    new StrictMode.ThreadPolicy.Builder().permitAll().build();
            StrictMode.setThreadPolicy(policy);
        }
    }
    public void listAllCameraSuport(){
        List<Camera.Size> tmpList = camera.getParameters().getSupportedPreviewSizes();
        for (int i=0;i<tmpList.size();i++)
        {
            System.out.println(tmpList.get(i).height + " " + tmpList.get(i).width);
        }
    }
    public void init(){
        surfaceView = (SurfaceView) findViewById(R.id.frame);
        imageView = (ImageView) findViewById(R.id.imageView);
        textViewName = (TextView) findViewById(R.id.textViewName);
        surfaceView.setMinimumHeight(surfaceView.getWidth());
        surfaceHolder = surfaceView.getHolder();
        surfaceHolder.addCallback((SurfaceHolder.Callback) this);
        surfaceHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        frame = new Camera.PreviewCallback() {
            @Override
            public void onPreviewFrame(byte[] data, Camera camera) {

                Camera.Parameters parameters = camera.getParameters();
                int width = parameters.getPreviewSize().width;
                int height = parameters.getPreviewSize().height;
                YuvImage yuv = new YuvImage(data, parameters.getPreviewFormat(), width, height, null);
                ByteArrayOutputStream out = new ByteArrayOutputStream();
                yuv.compressToJpeg(new Rect(0, 0, width, height), 100, out);
                byte[] bytes = out.toByteArray();
                final Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                bitmapCam = rotateBitmap(bitmap, 90);
                //Matrix matrix = new Matrix();
                //matrix.preScale(-1.0f, 1.0f);
                //bitmapCam = Bitmap.createBitmap(bitmapCam, 0, 0, bitmapCam.getWidth(), bitmapCam.getHeight(), matrix, true);


                try{
                    long start = System.currentTimeMillis();
                    JSONObject jsSend = new JSONObject();
                    Rect rect = new Rect();
                    rect.left = 210;
                    rect.right = 720 - 210;
                    rect.top = 210;
                    rect.bottom = 720 - 210;

                    bitmapCam = cropBitmap(bitmapCam, rect);
                    jsSend.accumulate("image", img2str(bitmapCam));

                    JSONObject jsRev = POST("http://ai.whis.tech/chatbot/recognition", jsSend);


                    long FPS = 1000/(System.currentTimeMillis() - start);
                    double conf = jsRev.getDouble("score");
                    String name = "";
                    if (conf>0.85) {

                        name = " Name: " + jsRev.getString("name");
                        name = name + "\n Confident: " + Double.toString(Math.round(conf * 100)) + "%";
                        name = name + "\n FPS: " + Long.toString(FPS);


                    } else {
                        name = " Name: " + "Unknow";
                        name = name + "\n Confident: 0" + "%";
                        name = name + "\n FPS: " + Long.toString(FPS);
                    }

                    textViewName.setText(name);
                    imageView.setImageBitmap(bitmapCam);
                }
                catch (Exception e){}
            }
        };
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        checkStrictMode();
        init();
        //loopForever();
    }

    public static JSONObject POST(String url, JSONObject jsSend) {
        InputStream inputStream = null;
        String result = "";
        try {
            // 1. create HttpClient

            HttpParams httpParameters = new BasicHttpParams();
            //int timeoutConnection = 3000;
            //HttpConnectionParams.setConnectionTimeout(httpParameters, timeoutConnection);
            //int timeoutSocket = 5000;
            //HttpConnectionParams.setSoTimeout(httpParameters, timeoutSocket);

            DefaultHttpClient httpClient = new DefaultHttpClient(httpParameters);

            HttpClient httpclient = new DefaultHttpClient(httpParameters);

            // 2. make POST request to the given URL
            HttpPost httpPost = new HttpPost(url);

            // 5. set json to StringEntity
            StringEntity se = new StringEntity(jsSend.toString());


            // 6. set httpPost Entity
            httpPost.setEntity(se);

            // 7. Set some headers to inform server about the type of the content
            httpPost.setHeader("Accept", "application/json");
            httpPost.setHeader("Content-type", "application/json");

            // 8. Execute POST request to the given URL
            HttpResponse httpResponse = httpclient.execute(httpPost);

            // 9. receive response as inputStream
            inputStream = httpResponse.getEntity().getContent();

            // 10. convert inputstream to string
            if (inputStream != null)
                result = convertInputStreamToString(inputStream);
            else
                result = "{}";
            return new JSONObject(result);

        } catch (Exception e) {
            System.out.println("loi" + e);
        }

        return null;
    }

    private static String convertInputStreamToString(InputStream inputStream) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        String line = "";
        String result = "";
        while ((line = bufferedReader.readLine()) != null)
            result += line;

        inputStream.close();
        return result;
    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int w, int h) {
        if (camera == null) {
            return;
        }
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
        //camera.stopPreview();
        //camera.release();

    }

    @Override
    public void surfaceCreated(SurfaceHolder hoder) {
        try {
            camera = Camera.open(Camera.CameraInfo.CAMERA_FACING_BACK);
        } catch (RuntimeException e) {
            System.err.println(e);
            return;
        }
        Camera.Parameters params = camera.getParameters();
        params.set("orientation", "portrait");
        camera.setDisplayOrientation(90);
        params.setRotation(90);
        params.setPreviewSize(720, 720);
        params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);




        listAllCameraSuport();
        System.out.println(Resources.getSystem().getDisplayMetrics().widthPixels);
        System.out.println(Resources.getSystem().getDisplayMetrics().heightPixels);


        camera.setParameters(params);

        camera.setPreviewCallback(frame);
        List<Camera.Size> sizes = params.getSupportedPictureSizes();
        Camera.Size msize = null;

        for (Camera.Size size : sizes) {
            msize = size;
        }

        params.setPictureSize(msize.width, msize.height);


        camera.setParameters(params);

        try {
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }
}
