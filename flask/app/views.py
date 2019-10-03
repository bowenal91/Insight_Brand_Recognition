from flask import render_template
from flask import request, url_for
from flaskexample import app
from detect import run_detector
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = "/home/alecbowen/Documents/Insight_Project/Multi_Class_Detection/Flask_App/FINAL/flaskexample/uploads/"
ALLOWED_EXTENSIONS = set(['mp4','avi'])


def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])
def upload_video():
    return render_template('upload_video.html')
@app.route('/static/', methods=['GET','POST'])

def result_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER,filename))
            run_detector(filename)
            return render_template('result_video.html')
            #return redirect(url_for('uploaded_file',filename=filename))
    return render_template('upload_video.html')

#@app.route('/go')
#def go():
#    query = request.args.get('query', '')
#    return render_template(
#        'go.html',
#        query=query,
#    )
