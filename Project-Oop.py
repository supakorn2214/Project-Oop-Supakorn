import random
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from random import randint
st.markdown(
    f"""
       <style>
       .stApp {{
           background-image: url("https://images.pexels.com/photos/1072179/pexels-photo-1072179.jpeg?blur=50");
           background-attachment: fixed;
           background-size: cover;
           /* opacity: 0.3; */
       }}
       </style>
       """,
    unsafe_allow_html=True
)

manu = ["Home", "ภาคเหนือ", "ภาคตะวันออกเฉียงเหนือ", "ภาคกลาง", "ภาคตะวันออก","ภาคตะวันตก", "ภาคใต้",
           "คำนวนความหนาแน่นของประชากรในพื้นที่"]
manu_bar = st.sidebar.selectbox("ภูมิภาค", manu)

if manu_bar == "Home":
    travel1 = st.sidebar
    if travel1:
        st.markdown(
            """
            <h1 style='text-align: center'>จำนวนประชากรแต่ละภาค และพื้นที่ของแต่ละจังหวัด ในประเทศไทย</h1>
            """,
            unsafe_allow_html=True
        )
        st.image("https://www.ggc.opm.go.th/upload/imagemap/big-map_55c1de003b8ec.png", width=700)
        st.text("...................................................................................\n")

        st.write(
            '&nbsp;&nbsp;&nbsp; จังหวัด เป็นเขตบริหารราชการส่วนภูมิภาคของประเทศไทย ปัจจุบันมีทั้งสิ้น 76 จังหวัด (ทั้งนี้ กรุงเทพมหานครไม่เป็นจังหวัด) '
            'จังหวัดถือเป็นระดับการปกครองของรัฐบาลลำดับแรก โดยเป็นหน่วยการปกครองส่วนภูมิภาคที่รวมท้องที่หลาย ๆ อำเภอเข้าด้วยกันและมีฐานะเป็นนิติบุคคล '
            'ในแต่ละจังหวัดปกครองด้วยผู้ว่าราชการจังหวัด')
        st.write(
            '&nbsp;&nbsp;&nbsp; การจัดแบ่งกลุ่มจังหวัดออกเป็นภาคต่าง ๆ มีการใช้เกณฑ์ที่แตกต่างกัน '
            'โดยมีทั้งการแบ่งอย่างเป็นทางการโดยราชบัณฑิตยสถานสำหรับใช้ในแบบเรียน '
            'และการแบ่งขององค์กรต่าง ๆ ตามแต่การใช้ประโยชน์ ชื่อของจังหวัดนั้นจะเป็นชื่อเดียวกับชื่ออำเภอที่เป็นที่ตั้งของศูนย์กลางจังหวัด เช่น '
            'ศูนย์กลางการปกครองของจังหวัดเพชรบุรีอยู่ที่อำเภอเมืองเพชรบุรี เป็นต้น '
            'แต่ชื่ออำเภอเหล่านี้มักเรียกย่อแต่เพียงว่า "อำเภอเมือง" ยกเว้นจังหวัดพระนครศรีอยุธยา ที่ใช้ชื่อจังหวัดเป็นชื่ออำเภอที่ตั้งศูนย์กลางการปกครองโดยตรง '
            '(อำเภอพระนครศรีอยุธยา)')
        st.write(
            '&nbsp;&nbsp;&nbsp; หน่วยการปกครองย่อยรองไปจากจังหวัดคือ "อำเภอ" ซึ่งมีทั้งสิ้น 878 อำเภอ ซึ่งจำนวนอำเภอนั้นจะแตกต่างกันไปในแต่ละจังหวัด '
            'ส่วนเขตการปกครองย่อยของกรุงเทพมหานครมีทั้งหมด 50 เขต ')
        st.text("...................................................................................\n")
        st.text("\n")

        st.markdown(
            """
            <h1 style='text-align: center'>ประวัติ</h1>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <h3 style='text-align:'>การปฏิรูปการบริหารราชการแผ่นดิน พ.ศ. 2435</h3>
            """,
            unsafe_allow_html=True
        )
        st.text("\n")
        st.write(
            '&nbsp;&nbsp;&nbsp; พ.ศ. 2435 พระบาทสมเด็จพระจุลจอมเกล้าเจ้าอยู่หัวโปรดเกล้าฯ ให้มีการปฏิรูประบบการบริหารราชการแผ่นดินครั้งใหญ่ '
            'ให้เป็นไปตามอย่างอารยประเทศในโลกตะวันตก โดยทรงตั้งกระทรวงขึ้นใหม่ 12 กระทรวง และโอนการปกครองหัวเมืองทั้งหมดให้มาขึ้นกับกระทรวงมหาดไทย '
            'โดยมีสมเด็จพระเจ้าบรมวงศ์เธอ กรมพระยาดำรงราชานุภาพ (ขณะนั้นดำรงพระยศเป็นกรมหมื่นดำรงราชานุภาพ) เป็นองค์ปฐมเสนาบดี')
        st.write(
            '&nbsp;&nbsp;&nbsp; เมื่องานการปกครองส่วนภูมิภาคขึ้นกับกระทรวงมหาดไทยแล้ว การจัดการปกครองด้วยระบบมณฑลเทศาภิบาลจึงได้เริ่มมีขึ้นในปี '
            'พ.ศ. 2437 โดยแบ่งระดับการปกครองจากสูงสุดไปหาต่ำสุดเป็นมณฑล, เมือง (เทียบเท่าจังหวัด), อำเภอ, ตำบล และหมู่บ้าน '
            'มีข้าหลวงเทศาภิบาลเป็นผู้กำกับดูแลมณฑล การก่อตั้งมณฑลนั้นจะเป็นไปตามลำดับโดยขึ้นอยู่กับความเป็นพื้นที่ที่มีความสำคัญในเชิงยุทธศาสตร์ด้วย '
            'วัตถุประสงค์สำคัญในการจัดการปกครองเช่นนี้ ก็เพื่อให้ส่วนกลางสามารถควบคุมดูแลหัวเมืองและจัดการผลประโยชน์แผ่นดินได้อย่างใกล้ชิด '
            'และลิดรอนอำนาจและอิทธิพลของเจ้าเมืองในระบบเดิมลงอย่างสิ้นเชิง')
        st.write(
            '&nbsp;&nbsp;&nbsp; การเปลี่ยนแปลงระบบการปกครองดังกล่าว ทำให้ขุนนางท้องถิ่นที่ต้องการรักษาฐานอำนาจและอิทธิพลของตนไว้ '
            'ก่อการกบฏต่อต้านอำนาจรัฐในบางภูมิภาค เหตุการณ์กบฏครั้งสำคัญคือกบฏผู้มีบุญอีสาน (หรือ "กบฏผีบาปผีบุญ") ซึ่งเกิดขึ้นเมื่อ พ.ศ. 2445'
            ' โดยอาศัยความเชื่อเรื่องยุคพระศรีอาริยเมตไตรย เป็นเครื่องมือในการปลุกระดมประชาชนให้ต่อต้านอำนาจรัฐ ขบวนการผู้มีบุญได้เคลื่อนไหวตามทั่วภาคอีสาน '
            'แต่ที่เป็นเหตุใหญ่ที่สุดอยู่ในพื้นที่จังหวัดอุบลราชธานี ซึ่งกลุ่มกบฏได้ก่อการถึงขั้นเผาเมืองเขมราฐและบังคับให้เจ้าเมืองเขมราฐร่วมขบวนการ '
            'แต่ที่สุดแล้วกบฏครั้งนี้ก็ถูกปราบปรามลงในเวลาไม่กี่เดือนต่อมา')
        st.write(
            '&nbsp;&nbsp;&nbsp; หลังปี พ.ศ. 2459 คำว่า "จังหวัด" ได้กลายเป็นคำที่เรียกหน่วยการปกครองระดับต่ำกว่ามณฑลแทนคำว่า "เมือง" '
            'เพื่อแยกความกำกวมจากคำว่าเมืองที่ใช้เรียกที่ตั้งศูนย์กลางการปกครองของจังหวัด (อำเภอเมือง)')
        st.write(
            '&nbsp;&nbsp;&nbsp; เมื่อสมเด็จพระเจ้าบรมวงศ์เธอ กรมพระยาดำรงราชานุภาพ ทรงลาออกจาตำแหน่งเสนาบดีกระทรวงมหาดไทยในปี พ.ศ. 2458 นั้น '
            'ประเทศสยามได้แบ่งการปกครองออกเป็น 19 มณฑล 72 จังหวัด ทั้งนี้ ได้รวมถึงจังหวัดพระนคร ซึ่งอยู่ในความรับผิดชอบของกระทรวงนครบาลจนถึง '
            'พ.ศ. 2465')
        st.write(
            '&nbsp;&nbsp;&nbsp; เดือนธันวาคม พ.ศ. 2458 พระบาทสมเด็จพระมงกุฎเกล้าเจ้าอยู่หัว โปรดเกล้าฯ ให้จัดตั้งหน่วยการปกครองระดับ "ภาค"'
            ' ขึ้นเพื่อกำกับมณฑล โดยมีผู้ปกครองภาคเรียกว่า "อุปราช" ในระยะนี้ได้มีการตั้งมณฑลต่าง ๆ เพิ่มขึ้นจนถึงปี พ.ศ. 2465 อีก 4 มณฑล '
            'แต่มณฑลดังกล่าวก็ถูกยุบลงในปี พ.ศ. 2468 และมีอีกหลายมณฑลที่ถูกยุบรวมในช่วงเวลาเดียวกัน เพื่อลดรายจ่ายของรัฐบาล '
            'อันเนื่องมาจากปัญหาเศรษฐกิจตกต่ำทั่วโลกในช่วงหลังสงครามโลกครั้งที่ 1')
        st.text("...................................................................................\n")
        st.markdown(
            """
            <h3 style='text-align:'>สมัยหลังเปลี่ยนแปลงการปกครอง</h3>
            """,
            unsafe_allow_html=True
        )
        st.text("\n")
        st.write(
            '&nbsp;&nbsp;&nbsp; หลังการปฏิวัติสยาม พ.ศ. 2475 ระบบมณฑลเทศาภิบาลถูกยกเลิกในปี พ.ศ. 2476 '
            'ทำให้จังหวัดกลายเป็นหน่วยการปกครองส่วนภูมิภาคระดับสูงสุด และตั้งแต่หลัง พ.ศ. 2500 เป็นต้นมา ก็ได้มีการจัดตั้งจังหวัดเพิ่มเติมอีกหลายแห่ง '
            'โดยการตัดแบ่งอาณาเขตบางส่วนจากจังหวัดที่มีขนาดใหญ่กว่า เช่น ในปี พ.ศ. 2515 ได้มีการแบ่งพื้นที่บางส่วนของจังหวัดอุบลราชธานีมาจัดตั้งเป็นจังหวัดยโสธร '
            'ในปี พ.ศ. 2520 มีการแบ่งพื้นที่บางส่วนของจังหวัดเชียงรายมาจัดตั้งเป็นจังหวัดพะเยา ในปี พ.ศ. 2525 ได้มีการแบ่งพื้นที่บางส่วนของจังหวัดนครพนมมาต'
            'ตั้งเป็นจังหวัดจังหวัดมุกดาหาร ในปี พ.ศ. 2536 ได้มีการตั้งจังหวัดขึ้นมาพร้อมกัน 3 จังหวัดคือจังหวัดหนองบัวลำภู (แยกจากจังหวัดอุดรธานี) จังหวัดสระแก้ว'
            '(แยกจากจังหวัดปราจีนบุรี) และจังหวัดอำนาจเจริญ (แยกจากจังหวัดอุบลราชธานี) และจังหวัดล่าสุดของประเทศไทยคือจังหวัดบึงกาฬ '
            '(แยกจากจังหวัดหนองคาย)ในปี พ.ศ. 2554')
        st.write(
            '&nbsp;&nbsp;&nbsp; ในปี พ.ศ. 2514 ได้มีการรวมจังหวัดพระนครและจังหวัดธนบุรีขึ้นเป็นเขตปกครองรูปแบบพิเศษชื่อ "นครหลวงกรุงเทพธนบุรี"'
            'และได้เปลี่ยนชื่อเป็น "กรุงเทพมหานคร" เมื่อ พ.ศ. 2515 ซึ่งเป็นการรวมภารกิจในการปกครองของทั้งสองจังหวัดในรูปแบบเทศบาลเข้าไว้ด้วยกัน '
            'ที่มาของผู้ว่าราชการกรุงเทพมหานครซึ่งเป็นผู้ปกครองสูงสุดของกรุงเทพมหานครนั้นมาจากการเลือกตั้ง ไม่ใช่การแต่งตั้งจากกระทรวงมหาดไทยอย่าง'
            'ผู้ว่าราชการจังหวัด')

        st.text("...................................................................................\n")
        st.markdown(
            """
            <h3 style='text-align:'>ลำดับเหตุการณ์</h3>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2469</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดกระบินทร์บุรี รวมเข้ากับจังหวัดปราจีนบุรี')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2474</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดสุโขทัย รวมเข้ากับจังหวัดสวรรคโลก')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดหล่มศักดิ์ รวมเข้ากับจังหวัดเพชรบูรณ์')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดธัญญบุรี รวมเข้ากับจังหวัดปทุมธานี')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดกาฬสินธุ์ รวมเข้ากับจังหวัดมหาสารคาม')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดหลังสวน รวมเข้ากับจังหวัดชุมพร')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดตะกั่วป่า รวมเข้ากับจังหวัดพังงา')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดสายบุรี รวมเข้ากับจังหวัดปัตตานีและนราธิวาส')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดพระประแดง รวมเข้ากับจังหวัดสมุทรปราการและธนบุรี')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดมีนบุรี รวมเข้ากับจังหวัดพระนครและฉะเชิงเทรา')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2489</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดสมุทรปราการ โดยแยกจากจังหวัดพระนคร')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดนนทบุรี โดยแยกจากจังหวัดธนบุรี')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดสมุทรสาคร โดยแยกจากจังหวัดธนบุรี')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดนครนายก โดยแยกจากจังหวัดปราจีนบุรีและสระบุรี')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2490</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดกาฬสินธุ์ โดยแยกจากจังหวัดมหาสารคาม')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2514</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ยุบเลิกจังหวัดพระนครและจังหวัดธนบุรี โดยจัดตั้งเป็นเขตปกครองรูปแบบพิเศษ นครหลวงกรุงเทพธนบุรี'
                 ' (ปัจจุบันคือกรุงเทพมหานคร)')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2515</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดยโสธร โดยแยกจากจังหวัดอุบลราชธานี')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2520</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดพะเยา โดยแยกจากจังหวัดเชียงราย')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2525</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดมุกดาหาร โดยแยกจากจังหวัดนครพนม')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2536</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดสระแก้ว โดยแยกจากจังหวัดปราจีนบุรี')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดอำนาจเจริญ โดยแยกจากจังหวัดอุบลราชธานี')
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดหนองบัวลำภู โดยแยกจากจังหวัดอุดรธานี')

        st.markdown("""<h5 style='text-align:&nbsp;'>พ.ศ. 2554</h5>""", unsafe_allow_html=True)
        st.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; จัดตั้งจังหวัดบึงกาฬ โดยแยกจากจังหวัดหนองคาย')
        st.text("...................................................................................\n")




if manu_bar == "ภาคเหนือ":
    travel2 = st.sidebar
    if travel2:
        st.header('ภาคเหนือ\n')
        st.title('1. เชียงราย (Chiang Rai)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/4/42/%E0%B8%98%E0%B8%87%E0%B8%88%E0%B8%B1%E0%B8%87%E0%B8%AB%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B9%80%E0%B8%8A%E0%B8%B5%E0%B8%A2%E0%B8%87%E0%B8%A3%E0%B8%B2%E0%B8%A2.jpg?20170621112945\n", width=700,)
        st.text('มีจำนวนประชากร 1,287,615 คน และมีพื้นที่ 11,678.4 ตร.กม.\n')

        st.title('2. เชียงใหม่ (Chiang Mai)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Flag_of_Chiang_Mai_Province.gif/1200px-Flag_of_Chiang_Mai_Province.gif\n",width=700)
        st.text('มีจำนวนประชากร 1,640,479 คน และมีพื้นที่ 20,107.0 ตร.กม.\n')

        st.title('3. น่าน (Nan)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c2/Flag_of_Nan_Province.jpg\n",width=700)
        st.text('มีจำนวนประชากร 476,363 คน และมีพื้นที่ 11,472.1 ตร.กม.\n')

        st.title('4. พะเยา (Phayao)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Flag_of_Phayao_Province.jpg/1200px-Flag_of_Phayao_Province.jpg\n",width=700)
        st.text('มีจำนวนประชากร 486,304 คน และมีพื้นที่ 6,335.1 ตร.กม.\n')

        st.title('5. แพร่ (Phrae)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/f4/%E0%B8%98%E0%B8%87%E0%B9%81%E0%B8%9E%E0%B8%A3%E0%B9%88.png\n",width=700)
        st.text('มีจำนวนประชากร 447,564 คน และมีพื้นที่ 6,538.6 ตร.กม.\n')

        st.title('6. แม่ฮ่องสอน (Mae Hong Son)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Flag_Mae_Hong_Son_Province.png/1200px-Flag_Mae_Hong_Son_Province.png\n",width=700)
        st.text('มีจำนวนประชากร 242,742 คน และมีพื้นที่ 12,681.3 ตร.กม.\n')

        st.title('7. ลำปาง (Lampang)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Flag_of_Lampang_Province.png/1200px-Flag_of_Lampang_Province.png\n",width=700)
        st.text('มีจำนวนประชากร 761,949 คน และมีพื้นที่ 12,534.0 ตร.กม.\n')

        st.title('8. ลำพูน (Lamphun)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Lamphun_provincial_flag.png/1200px-Lamphun_provincial_flag.png\n",width=700)
        st.text('มีจำนวนประชากร 404,560 คน และมีพื้นที่ 4,505.9 ตร.กม.\n')

        st.title('9. อุตรดิตถ์ (Uttaradit)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/Uttaradit_provincial_flag.png/1200px-Uttaradit_provincial_flag.png\n",width=700)
        st.text('มีจำนวนประชากร 462,618 คน และมีพื้นที่ 7,838.6 ตร.กม.\n')

if manu_bar == "ภาคตะวันออกเฉียงเหนือ":
    travel3 = st.sidebar
    if travel3:
        st.header('ภาคตะวันออกเฉียงเหนือ\n')
        st.title('1. กาฬสินธุ์ (Kalasin)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Flag_Karasin_Province.png/1200px-Flag_Karasin_Province.png\n",width=700)
        st.text('มีจำนวนประชากร 982,578 คน และมีพื้นที่ 6,946.7 ตร.กม.\n')

        st.title('2. ขอนแก่น (Khon Kaen)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/52/Khon_Kaen_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 1,767,601 คน และมีพื้นที่ 10,886.0 ตร.กม.\n')

        st.title('3. ชัยภูมิ (Chaiyaphum)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/0d/Chaiyaphum_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 1,127,423 คน และมีพื้นที่ 12,778.3 ตร.กม.\n')

        st.title('4. นครพนม (Nakhon Phanom)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/Flag_of_Nakhon_Phanom_Province.svg/1200px-Flag_of_Nakhon_Phanom_Province.svg.png\n", width=700)
        st.text('มีจำนวนประชากร 703,392 คน และมีพื้นที่ 5,512.7 ตร.กม.\n')

        st.title('5. นครราชสีมา (Nakhon Ratchasima)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/%E0%B8%88%E0%B8%B1%E0%B8%87%E0%B8%AB%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AA%E0%B8%B5%E0%B8%A1%E0%B8%B2.jpg/1200px-%E0%B8%88%E0%B8%B1%E0%B8%87%E0%B8%AB%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B8%99%E0%B8%84%E0%B8%A3%E0%B8%A3%E0%B8%B2%E0%B8%8A%E0%B8%AA%E0%B8%B5%E0%B8%A1%E0%B8%B2.jpg\n", width=700)
        st.text('มีจำนวนประชากร 2,628,818 คน และมีพื้นที่ 20,494.0 ตร.กม.\n')

        st.title('6. บึงกาฬ (Bueng Kan)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/96/Flag_of_Bueng_Kan_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 385,053 คน และมีพื้นที่ 4,305 ตร.กม.\n')

        st.title('7. บุรีรัมย์ (Buri Ram )\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/ea/Flag_of_Buriram_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 1,553,765 คน และมีพื้นที่ 10,322.9 ตร.กม.\n')

        st.title('8. มหาสารคาม (Maha Sarakham)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/79/Mahasarakham_PV_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 940,911 คน และมีพื้นที่ 5,291.7 ตร.กม.\n')

        st.title('9. มุกดาหาร (Mukdahan)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/93/Mukdahan_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 339,575 คน และมีพื้นที่ 4,339.8 ตร.กม.\n')

        st.title('10. ยโสธร (Yasothon)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/b9/Yasothon_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 539,542 คน และมีพื้นที่ 4,161.7 ตร.กม.\n')

        st.title('11. ร้อยเอ็ด (Roi Et)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/2/2a/%E0%B8%98%E0%B8%87%E0%B8%A3%E0%B9%89%E0%B8%AD%E0%B8%A2%E0%B9%80%E0%B8%AD%E0%B9%87%E0%B8%94.gif?20130311193322\n", width=700)
        st.text('มีจำนวนประชากร 1,309,708 คน และมีพื้นที่ 8,299.4 ตร.กม.\n')

        st.title('12. เลย (Loei)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Flag_of_Loei_Province.png/1200px-Flag_of_Loei_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 624,066 คน และมีพื้นที่ 11,424.6 ตร.กม.\n')

        st.title('13. ศรีสะเกษ (Si Sa Ket)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e4/Sisaket_drapeau.gif\n", width=700)
        st.text('มีจำนวนประชากร 1,452,471 คน และมีพื้นที่ 8,840.0 ตร.กม.\n')

        st.title('14. สกลนคร (Sakon Nakhon)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Sakon_Nakhon_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 1,122,905 คน และมีพื้นที่ 9,605.8 ตร.กม.\n')

        st.title('15. สุรินทร์ (Surin)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/05/Flag_of_Surin_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 1,381,761 คน และมีพื้นที่ 8,124.1 ตร.กม.\n')

        st.title('16. หนองคาย (Nong Khai)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Flag_Nong_Khai_Province.png/1200px-Flag_Nong_Khai_Province.png\n",width=700)
        st.text('มีจำนวนประชากร 509,395 คน และมีพื้นที่ 3,027.0 ตร.กม.\n')

        st.title('17. หนองบัวลำภู (Nong Bua Lam Phu)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Flag_of_Nong_Bua_Lamphu_Province.jpeg/1200px-Flag_of_Nong_Bua_Lamphu_Province.jpeg\n", width=700)
        st.text('มีจำนวนประชากร 502,868 คน และมีพื้นที่ 3,859.0 ตร.กม.\n')

        st.title('18. อุดรธานี (Udon Thani)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Flag_of_Udon_Thani_Province.jpg/1200px-Flag_of_Udon_Thani_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 1,544,786 คน และมีพื้นที่ 11,730.3 ตร.กม.\n')

        st.title('19. อุบลราชธานี (Ubon Ratchathani)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Ubon_Ratchathani_Province_Flags.svg/1200px-Ubon_Ratchathani_Province_Flags.svg.png\n", width=700)
        st.text('มีจำนวนประชากร 1,813,088 คน และมีพื้นที่ 15,744.8 ตร.กม.\n')

        st.title('20. อำนาจเจริญ (Amnat Charoen)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Flag_of_Amnat_Charoen_Province.png/1200px-Flag_of_Amnat_Charoen_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 372,137 คน และมีพื้นที่ 3,161.2 ตร.กม.\n')

if manu_bar == "ภาคกลาง":
    travel4 = st.sidebar
    if travel4:
        st.header('ภาคกลาง\n')
        st.title('1. กรุงเทพมหานคร (Bangkok)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Flag_of_Bangkok.svg/1200px-Flag_of_Bangkok.svg.png\n",width=700)
        st.text('มีจำนวนประชากร 5,682,415 คน และมีพื้นที่ 1,568.7 ตร.กม.\n')

        st.title('2. กำแพงเพชร (Kamphaeng Phet)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/8f/Flag_of_Kamphaeng_Phet_Province.gif\n", width=700)
        st.text('มีจำนวนประชากร 729,133 คน และมีพื้นที่ 8,607.5 ตร.กม.\n')

        st.title('3. ชัยนาท (Chai Nat)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/bf/Chai_Nat_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 329,722 คน และมีพื้นที่ 2,469.7 ตร.กม.\n')

        st.title('4. นครนายก (Nakhon Nayok)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Flag_Nakhon_Nayok_Province.png/1200px-Flag_Nakhon_Nayok_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 259,342 คน และมีพื้นที่ 2,122.0 ตร.กม.\n')

        st.title('5. นครปฐม (Nakhon Pathom)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Flag_of_Nakhon_Pathom_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 911,492 คน และมีพื้นที่ 2,168.3 ตร.กม.\n')

        st.title('6. นครสวรรค์ (Nakhon Sawan)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Nakhon_Sawan_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 1,065,334 คน และมีพื้นที่ 9,597.7 ตร.กม.\n')

        st.title('7. นนทบุรี (Nonthaburi)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/%E0%B8%98%E0%B8%87%E0%B8%88%E0%B8%B1%E0%B8%87%E0%B8%AB%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B8%99%E0%B8%99%E0%B8%97%E0%B8%9A%E0%B8%B8%E0%B8%A3%E0%B8%B5.svg/768px-%E0%B8%98%E0%B8%87%E0%B8%88%E0%B8%B1%E0%B8%87%E0%B8%AB%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B8%99%E0%B8%99%E0%B8%97%E0%B8%9A%E0%B8%B8%E0%B8%A3%E0%B8%B5.svg.png\n", width=700)
        st.text('มีจำนวนประชากร 1,229,735 คน และมีพื้นที่ 622.3 ตร.กม.\n')


        st.title('8. ปทุมธานี (Pathum Thani)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/36/Pathum_Thani_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 1,129,115 คน และมีพื้นที่ 1,525.9 ตร.กม.\n')

        st.title('9. พระนครศรีอยุธยา (Phra Nakhon Si Ayutthaya)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Flag_of_Phra_Nakhon_Si_Ayutthaya_Province.svg/1200px-Flag_of_Phra_Nakhon_Si_Ayutthaya_Province.svg.png\n", width=700)
        st.text('มีจำนวนประชากร 808,360 คน และมีพื้นที่ 2,556.6 ตร.กม.\n')

        st.title('10. พิจิตร (Phichit)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/82/Flag_of_Phichit_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 541,868 คน และมีพื้นที่ 4,531.0 ตร.กม.\n')

        st.title('11. พิษณุโลก (Phitsanulok)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Pitsanulok_flag.svg/768px-Pitsanulok_flag.svg.png\n", width=700)
        st.text('มีจำนวนประชากร 813,852 คน และมีพื้นที่ 10,815.8 ตร.กม.\n')

        st.title('12. เพชรบูรณ์ (Phetchabun)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Phetchabun_provincial_flag.png/1200px-Phetchabun_provincial_flag.png\n", width=700)
        st.text('มีจำนวนประชากร 996,031 คน และมีพื้นที่ 12,668.4 ตร.กม.\n')

        st.title('13. ลพบุรี (Lop Buri)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Flag_Lop_Buri_Province.png/1200px-Flag_Lop_Buri_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 755,854 คน และมีพื้นที่ 6,199.8 ตร.กม.\n')

        st.title('14. สมุทรปราการ (Samut Prakan)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/3/3b/%E0%B8%98%E0%B8%87%E0%B8%AA%E0%B8%A1%E0%B8%B8%E0%B8%97%E0%B8%A3%E0%B8%9B%E0%B8%A3%E0%B8%B2%E0%B8%81%E0%B8%B2%E0%B8%A3.gif?20130311185900\n", width=700)
        st.text('มีจำนวนประชากร 1,310,766 คน และมีพื้นที่ 1,004.1 ตร.กม.\n')

        st.title('15. สมุทรสงคราม (Samut Songkhram)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Flag_Samut_Songkhram_Province.png/1200px-Flag_Samut_Songkhram_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 194,057 คน และมีพื้นที่ 416.7 ตร.กม.\n')

        st.title('16. สมุทรสาคร (Samut Sakhon)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Flag_Samut_Sakhon_Province.png/1200px-Flag_Samut_Sakhon_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 491,887 คน และมีพื้นที่ 872.3 ตร.กม.\n')

        st.title('17. สระบุรี (Saraburi)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Provincial_Flag_of_Saraburi.svg/1200px-Provincial_Flag_of_Saraburi.svg.png\n", width=700)
        st.text('มีจำนวนประชากร 617,384 คน และมีพื้นที่ 3,576.5 ตร.กม.\n')

        st.title('18. สิงห์บุรี (Sing Buri)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Flag_of_Sing_Buri_Province.png/1200px-Flag_of_Sing_Buri_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 214,661 คน และมีพื้นที่ 822.5 ตร.กม.\n')

        st.title('19. สุโขทัย (Sukhothai)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Flag_of_Sukhothai_Province.svg/1200px-Flag_of_Sukhothai_Province.svg.png\n", width=700)
        st.text('มีจำนวนประชากร 601,778 คน และมีพื้นที่ 6,596.1 ตร.กม.\n')

        st.title('20. สุพรรณบุรี (	Suphan Buri)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/4a/Flag_of_Suphan_Buri_%28province%29.gif\n", width=700)
        st.text('มีจำนวนประชากร 845,950 คน และมีพื้นที่ 5,358.0 ตร.กม.\n')

        st.title('21. อ่างทอง (Ang Thong)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7a/Ang_Thong_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 284,970 คน และมีพื้นที่ 968.4 ตร.กม.\n')

        st.title('22. อุทัยธานี (Uthai Thani)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a6/Flag_of_Uthai_Thani_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 327,959 คน และมีพื้นที่ 6,730.3 ตร.กม.\n')

if manu_bar == "ภาคตะวันออก":
    travel5 = st.sidebar
    if travel5:
        st.header('ภาคตะวันออก\n')
        st.title('1. จันทบุรี (Chanthaburi)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c9/Flag_of_Chanthaburi_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 514,616 คน และมีพื้นที่ 6,338.0 ตร.กม.\n')

        st.title('2. ฉะเชิงเทรา (Chachoengsao)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Flag_of_Chachoengsao_Province.png/1200px-Flag_of_Chachoengsao_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 673,933 คน และมีพื้นที่ 5,351.0 ตร.กม.\n')

        st.title('3. ชลบุรี (Chon Buri)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_Chon_Buri_Province.png/1200px-Flag_Chon_Buri_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 1,509,125 คน และมีพื้นที่ 4,363.0 ตร.กม.\n')

        st.title('4. ตราด (Trat)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/5e/Flag_of_Trat_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 220,921 คน และมีพื้นที่ 2,819.0 ตร.กม.\n')

        st.title('5. ปราจีนบุรี (Prachin Buri)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Flag_of_Prachin_Buri_Province.jpg/1200px-Flag_of_Prachin_Buri_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 466,572 คน และมีพื้นที่ 4,762.4 ตร.กม.\n')

        st.title('6. ระยอง (Rayong)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/b/b0/Flag_of_Rayong.jpg?20171207104905\n", width=700)
        st.text('มีจำนวนประชากร 626,402 คน และมีพื้นที่ 3,552.0 ตร.กม.\n')

        st.title('7. สระแก้ว (Sa Kaeo)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/b/b9/Flag_of_Sa_Kaeoi.jpg?20171207104955\n", width=700)
        st.text('มีจำนวนประชากร 485,632 คน และมีพื้นที่ 7,195.1 ตร.กม.\n')

if manu_bar == "ภาคตะวันตก":
    travel6 = st.sidebar
    if travel6:
        st.header('ภาคตะวันตก\n')
        st.title('1. กาญจนบุรี (Kanchanaburi)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/59/Flag_of_Kanchanaburi_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 839,776 คน และมีพื้นที่ 19,483.2 ตร.กม.\n')

        st.title('2. ตาก (Tak)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c2/Flag_of_Tak_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 525,684 คน และมีพื้นที่ 16,406.6 ตร.กม.\n')

        st.title('3. ประจวบคีรีขันธ์ (Prachuap Khiri Khan)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/94/Flag_of_Prachuap_Khiri_Khan_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 509,134 คน และมีพื้นที่ 6,367.6 ตร.กม.\n')

        st.title('4. เพชรบุรี (Phetchaburi)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Flag_Petchaburi_Province.png/1200px-Flag_Petchaburi_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 464,033 คน และมีพื้นที่ 6,225.1 ตร.กม.\n')

        st.title('5. ราชบุรี (Ratchaburi)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/0f/Ratchaburi_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 839,075 คน และมีพื้นที่ 5,196.5 ตร.กม.\n')

if manu_bar == "ภาคใต้":
    travel7 = st.sidebar
    if travel7:
        st.header('ภาคใต้\n')
        st.title('1. กระบี่ (Krabi)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Flag_of_Krabi_Province.jpg/1200px-Flag_of_Krabi_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 469,769 คน และมีพื้นที่ 4,708.5 ตร.กม.\n')

        st.title('2. ชุมพร (Chumphon)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Chumphon_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 509,650 คน และมีพื้นที่ 6,010.8 ตร.กม.\n')

        st.title('3. ตรัง (	Trang)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Trang_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 643,072 คน และมีพื้นที่ 4,917.5 ตร.กม.\n')

        st.title('4. นครศรีธรรมราช (Nakhon Si Thammarat)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Flag_of_Nakhon_Si_Thammarat_Province.jpg/1200px-Flag_of_Nakhon_Si_Thammarat_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 1,557,482 คน และมีพื้นที่ 9,942.5 ตร.กม.\n')

        st.title('5. นราธิวาส (Narathiwat)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/1/17/%E0%B8%98%E0%B8%87%E0%B8%99%E0%B8%A3%E0%B8%B2%E0%B8%98%E0%B8%B4%E0%B8%A7%E0%B8%B2%E0%B8%AA.gif?20130311191833\n", width=700)
        st.text('มีจำนวนประชากร 796,239 คน และมีพื้นที่ 4,475.4 ตร.กม.\n')

        st.title('6. ปัตตานี (Pattani)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/3/3e/Pattani_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 709,796 คน และมีพื้นที่ 1,940.4 ตร.กม.\n')

        st.title('7. พังงา (Phangnga)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7c/Phangnga_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 267,491 คน และมีพื้นที่ 4,170.9 ตร.กม.\n')

        st.title('8. พัทลุง (Phatthalung)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/2/27/Phatthalung_Provincial_flag.jpg?20160517140851\n", width=700)
        st.text('มีจำนวนประชากร 524,857 คน และมีพื้นที่ 3,424.5 ตร.กม.\n')

        st.title('9. ภูเก็ต (Phuket)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/ff/Phuket_Flag.png\n", width=700)
        st.text('มีจำนวนประชากร 402,017 คน และมีพื้นที่ 543.0 ตร.กม.\n')

        st.title('10. ยะลา (Yala)\n')
        st.image("https://upload.wikimedia.org/wikipedia/th/2/28/Yala_prov_TH.gif?20160222104927\n", width=700)
        st.text('มีจำนวนประชากร 527,295 คน และมีพื้นที่ 4,521.1 ตร.กม.\n')

        st.title('11. ระนอง (Ranong)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Flag_Ranong_Province.png/1200px-Flag_Ranong_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 190,399 คน และมีพื้นที่ 3,298.0 ตร.กม.\n')

        st.title('12. สตูล (Satun)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/28/Flag_of_Satun_Province.jpg\n", width=700)
        st.text('มีจำนวนประชากร 319,700 คน และมีพื้นที่ 2,479.0 ตร.กม.\n')

        st.title('13. สงขลา (Songkhla)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Flag_Songkhla_Province.png/1200px-Flag_Songkhla_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 1,424,230 คน และมีพื้นที่ 7,393.9 ตร.กม.\n')

        st.title('14. สุราษฎร์ธานี (Surat Thani)\n')
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Flag_Surat_Thani_Province.png/1200px-Flag_Surat_Thani_Province.png\n", width=700)
        st.text('มีจำนวนประชากร 1,057,581 คน และมีพื้นที่ 12,891.5 ตร.กม.\n')

if manu_bar == "คำนวนความหนาแน่นของประชากรในพื้นที่":
    travel8 = st.sidebar
    if travel8:
        st.markdown(
            """
            <h1 style='text-align: center'>โปรแกรมคำนวนประชากรในพื้นที่</h1>
            """,
            unsafe_allow_html=True
        )
        population = st.number_input("Population : ",)
        area = st.number_input("Area :")
        x2 = st.button("calculate")
        if x2:
            calculator = st.write(f'จำนวนความหนาแน่นของประชากรในพื้นที่  : {population / area:.2f} ')
            st.write("โดยใช้ จำนวนประชากร / พื้นที่ (ตร.กม.)")

        def load_house_data():
            return pd.read_excel('ProJect-Oop.xlsx')

        def save_model(model):
            joblib.dump(model, 'model.joblib')

        def load_model():
            return joblib.load('model.joblib')

        def generate_wedtest_data():
            rng = np.random.RandomState(0)
            n = 77
            n1 = random.randrange(200000, 2000000)
            n2 = random.randrange(1000, 20000)
            x = np.round(n1 * rng.rand(n))  # จำนวนประชากร
            y = np.round(n2 * rng.rand(n))  # พื้นที่
            df = pd.DataFrame({
                'Area': y,
                'Population': x
            })
            df.to_excel('ProJect-Oop.xlsx')

        generateb = st.button('generate ProJect-Oop.xlsx')
        if generateb:
            st.write('generating "ProJect-Oop.xlsx" ...')
            generate_wedtest_data()

        loadb = st.button('load ProJect-Oop.xlsx')
        if loadb:
            st.write('loading "ProJect-Oop.xlsx ..."')
            df = pd.read_excel('ProJect-Oop.xlsx', index_col=0)
            st.dataframe(df)
            fig, ax = plt.subplots()
            df.plot.scatter(x='Area', y='Population',ax=ax)
            st.pyplot(fig)

        trainb = st.button('train Program')
        if trainb:
            st.write('training model ...')
            df = pd.read_excel('ProJect-Oop.xlsx', index_col=0)
            model = LinearRegression()
            st.dataframe(df)
            save_model(model)
            # &#127481 &#127469

        chart = st.button('Chart Program')
        if chart:
            tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
            df = pd.read_excel('ProJect-Oop.xlsx', index_col=0)
            tab1.line_chart(df)
            tab2.write(df)
