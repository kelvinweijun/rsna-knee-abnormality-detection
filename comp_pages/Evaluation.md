Submissions are evaluated by the average [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted confidence scores and the observed targets across the twelve targets:

$$
\text{Final Score} = \frac{1}{12}\sum_{i=0}^{11} AUC_i
$$

The final score is, in other words, the macro-averaged AUC ROC.

## Submission File
For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
...
```