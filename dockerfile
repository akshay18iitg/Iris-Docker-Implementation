FROM continuumio/anaconda3
COPY ./iris  /usr/local/python/
EXPOSE 5005
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt
CMD python app.py