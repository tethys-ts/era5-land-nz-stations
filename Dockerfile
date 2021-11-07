FROM tethysts/tethys-extraction-base:1.4

# ENV TZ='Pacific/Auckland'

# RUN apt-get update && apt-get install -y unixodbc-dev gcc g++ libspatialindex-dev python-rtree

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY precip.py ./

CMD ["python", "-u", "precip.py"]
