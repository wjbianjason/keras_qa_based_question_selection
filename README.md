# keras_qa_based_question_selection
Keras-Based LSTM Conv model for Question Selection
<br>
This code is remplementation of lstm+cnn of Improved Representation Learning for Question Answer Matching
<br>
You also can see that I utilize the gesd and weights shared method in APPLYING DEEP LEARNING TO ANSWER SELECTION:
A STUDY AND AN OPEN TASK
<br>
Dataset is InsuranceQA, Now precision in dev is 65% when epoch = 100
<br>
##Train Step
python main.py train
<br>
##Evaluate Step
python main.py evaluate 100
<br>
of course, you can change the second param which is epoch number.
