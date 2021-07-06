from fastai.vision.all import (
    load_learner,
    PILImage,
)
import glob
import streamlit as st
from PIL import Image
from random import shuffle
import urllib.request

# set images
image1 = Image.open('sidebar1.jpg')
image2 = Image.open('menu1.jpg')
image3 = Image.open('menu2.jpg')

#MODEL_URL = "https://github.com/GemmyTheGeek/FoodyDudy/raw/main/resnet34-10.pkl"
MODEL_URL = "http://freelyblog.com/resnet34-10.pkl"
urllib.request.urlretrieve(MODEL_URL, "model.pkl")
learn_inf = load_learner('model.pkl', cpu=True)

thaimenu=[
"แกงเขียวหวานไก่","แกงเทโพ","แกงเลียง","แกงจืดเต้าหู้หมูสับ","แกงจืดมะระยัดไส้",
"แกงมัสมั่นไก่","แกงส้มกุ้ง","ไก่ผัดเม็ดมะม่วงหิมพานต์","ไข่เจียว","ไข่ดาว",
"ไข่พะโล้","ไข่ลูกเขย","กล้วยบวชชี","ก๋วยเตี๋ยวคั่วไก่","กะหล่ำปลีผัดน้ำปลา",
"กุ้งแม่น้ำเผา","กุ้งอบวุ้นเส้น","ขนมครก","ข้าวเหนียวมะม่วง","ข้าวขาหมู",
"ข้าวคลุกกะปิ","ข้าวซอยไก่","ข้าวผัด","ข้าวผัดกุ้ง","ข้าวมันไก่",
"ข้าวหมกไก่","ต้มข่าไก่","ต้มยำกุ้ง","ทอดมัน","ปอเปี๊ยะทอด",
"ผักบุ้งไฟแดง","ผัดไท","ผัดกะเพรา","ผัดซีอิ๊วเส้นใหญ่","ผัดฟักทองใส่ไข่",
"ผัดมะเขือยาวหมูสับ","ผัดหอยลาย","ฝอยทอง","พะแนงไก่","ยำถั่วพู",
"ยำวุ้นเส้น","ลาบหมู","สังขยาฟักทอง","สาคูไส้หมู","ส้มตำ","หมูปิ้ง","หมูสะเต๊ะ","ห่อหมก"
]

engmenu=[
"Chicken Green Curry","Pork Curry With Morning Glory","Spicy Mixed Vegetable Soup","Pork Chopped Tofu Soup","Stuffed Bitter Gourd Broth",
"Chicken Mussaman Curry","Sour Soup","Stir-Fried Chicken With Chestnuts","Omelet","Fried Egg",
"Egg And Pork In Sweet Brown Sauce","Egg With Tamarind Sauce","Banana In Coconut Milk","Stir-Fried Rice Noodles With Chicken","Fried Cabbage With Fish Sauce",
"Grilled River Prawn","Baked Prawns With Vermicelli","Coconut Rice Pancake","Mango Sticky Rice","Thai Pork Leg Stew",
"Shrimp Paste Fried Rice","Curried Noodle Soup With Chicken","Fried Rice","Shrimp Fried Rice","Steamed Capon In Flavored Rice",
"Thai Chicken Biryani","Thai Chicken Coconut Soup","River Prawn Spicy Soup","Fried Fish-Paste Balls","Deep-Fried Spring Roll",
"Stir-Fried Chinese Morning Glory","Fried Noodle Thai Style With Prawns","Stir-Fried Thai Basil With Minced Pork","Fried Noodle In Soy Sauce","Stir-Fried Pumpkin With Eggs",
"Stir-Fried Eggplant With Soybean Paste Sauce","Stir-Fried Clams With Roasted Chili Paste","Golden Egg Yolk Threads","Chicken Panang ","Thai Wing Beans Salad",
"Spicy Glass Noodle Salad","Spicy Minced Pork Salad","Egg Custard In Pumpkin","Tapioca Balls With Pork Filling","Green Papaya Salad",
"Thai-Style Grilled Pork Skewers","Pork Satay With Peanut Sauce","Steamed Fish With Curry Paste"
]

def predict(img, learn):
    # make prediction
    pred, pred_idx, pred_prob = learn.predict(img)
    # Display the prediction
    thaimenuname = thaimenu[int(pred_idx)]
    engmenuname = engmenu[int(pred_idx)]
    st.success(f"This is {engmenuname} ({thaimenuname}) with the probability of {pred_prob[pred_idx]*100:.02f}%")
    # Display the test image
    # st.image(img, use_column_width=True)
    col1.image(img, use_column_width=True)
    # col2.image(''+image3)
    col2.image(Image.open('./thaimenu/'+str(int(pred_idx))+'.png'))

##################################
# Top Main
##################################
col1, col2 = st.beta_columns(2)

##################################
# Col1
##################################
col1.header("Your Food Image")

##################################
# Col2
##################################
col2.header("Ingredients and Spiciness")

st.sidebar.image(image1)

fname = st.sidebar.file_uploader('Enter thai food to classify',type=['png', 'jpg', 'jpeg'],accept_multiple_files=False)
if fname is None:
    # fname = valid_images[0]
    col1.image(image2)
    col2.image(image3)
else :
    img = PILImage.create(fname)
    predict(img, learn_inf)

##################################
# sidebar
##################################
st.sidebar.markdown('This app, Foody Dudy, was developed by Gemmy Somboontham and is a part of AI Builders Gen I, organized by VISTEC, Central Group and AIReaserch Thailand.')
st.sidebar.write("AI Builders page [link](https://www.facebook.com/aibuildersx)")
st.sidebar.write("FoodyDudy at Github [link](https://github.com/GemmyTheGeek/FoodyDudy)")
st.sidebar.write("Checkout Colab [link](https://colab.research.google.com/github/GemmyTheGeek/FoodyDudy/inferencer.ipynb)")
st.sidebar.write("Support Me! [link](https://www.youtube.com/c/codingforkids?sub_confirmation=1)")

st.text(' ')
st.text(' ')

my_expander = st.beta_expander(label='End Credits Scene')
with my_expander:
    '[P Charin](https://github.com/cstorm125/) : Thank You so much for your help, tips, and advice because if it weren’t for you, my model wouldn’t be the way it is today.'
    '[P Joy](https://th.linkedin.com/in/nattakarnphaphoom), [UncleEngineer](https://www.facebook.com/UncleEngineer), [P A](https://www.facebook.com/kanes.sumetpipat/) : Even though we didn’t talk too much during this program, I still am thankful for your incredible encouragement and support.'
    '[FastAI](https://course.fast.ai/) : Thank You, fastai, for giving me more knowledge on AI and Neural Networks; comparing my understanding of AI before I began the course compared to after the course is immeasurable.'
    'Google Colab : Thank you, Google team, for making Colab; because of Colab, I was able to code more smoothly since I am used to it.'
    'DuckDuckGo : Thank you, DuckDuckGo, for giving me most of my dataset for Foody Dudy.'
    'Heroku : Thankyou to Heroku for letting me load my app in a blink of an eye, and being free.'
    'Family : I also would like to thankyou my family for giving me support during this time.'
    'Stackoverflow : I cant live without you.'
