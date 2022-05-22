
import cv2
import numpy as np

def draw_results(input_image,det,output):
  font = cv2.FONT_HERSHEY_PLAIN

  labels = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
      'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
      'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
      'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
      'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
      'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
      'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
      'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
      'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
      'Wearing_Necktie', 'Young']
  position0  = np.where(output[0]>0.4)[0]
  count = 15
  for i2 in range(len(position0)):
    cv2.putText(input_image, str(labels[position0[i2]])+" "+str(np.round(output[0][position0[i2]],3)), (det.right(),15+count), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
  
  return input_image