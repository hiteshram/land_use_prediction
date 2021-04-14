import requests

#r=requests.get('https://maps.googleapis.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=13&size=600x300&maptype=roadmap&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318&markers=color:red%7Clabel:C%7C40.718217,-73.998284&key=')
r=requests.get('https://maps.googleapis.com/maps/api/staticmap?center=40.714728,-73.998672&zoom=20&size=400x400&maptype=satellite&key=AIzaSyA4y9y51IS_up6qEGih1jr2dbrhFhlSsAo')

with open(r'D:\University of Colorado\Subjects\Deep Learning\Project\land_use_prediction\temp.jpg','wb') as f:
    f.write(r.content)