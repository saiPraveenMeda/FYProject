from django.core import serializers
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, render_to_response, redirect
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import json,os
import subprocess
from .serializers import FileSerializer
from django.core.files.storage import FileSystemStorage
import base64
from django.views.decorators.csrf import csrf_exempt

def temp(request) :
    subprocess.Popen(['/home/sai/Desktop/project_temp/temp2.sh'], shell=True)
    result = 'Success'
    response = {}
    return render(request,'codeable/temp.djt',response)

@csrf_exempt
def FileView(request):
    folder='my_folder/' 
    if request.method == 'POST':
        temp = json.loads(request.body.decode("utf-8"))
        # myfile = request.FILES['file']
        # fs = FileSystemStorage(location=folder) #defaults to   MEDIA_ROOT  
        # filename = fs.save(myfile.name, myfile)
        # file_url = fs.url(filename)
        # return HttpResponse('file_url=' + file_url)

        myfiledata = temp['file']
       # print('file=' + myfiledata)
        imgdata = base64.b64decode(myfiledata)
        filename = 'temp1.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(imgdata) 
        #print('json=' + str(temp))
        subprocess.Popen(['/media/abhilash/0A129CCF129CC15D/project/process.sh'], shell=True)
        return HttpResponse('ok!!!')
    else:
         return HttpResponse('some error occured')