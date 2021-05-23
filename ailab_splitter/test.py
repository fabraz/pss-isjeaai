from nn_modules import *
from torchsummary import summary

'''model = TwoPagesEffModule(2)

summary(model, [(3, 224, 224), (3,224, 224)])


model = TwoPagesEffModule(2)

summary(model, [(3, 224, 224), (3,224, 224)])
'''
#model = OnePageEffModule(4, True, 12)

#summary(model, (3, 224, 224))#, (3,224, 224), (3,224, 224)])

#print(model.base_model)

model = TwoPagesVGG16Module(4)

summary(model, [(3, 224, 224),  (3,224, 224)])

#print(model.base_model)