from django.shortcuts import render
from .models import predict
# Create your views here.
def index(request):
    if request.method == 'POST':
        data=request.POST['Nodes']
        result=predict(data)
        return render(request,"index.html",{"predict":result})
    else:
        return render(request,"index.html",{})