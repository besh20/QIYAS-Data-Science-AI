# RAG Retrieval -- Evaluation

- Top-1 accuracy: **8/10 = 80%**
- Top-3 accuracy: **9/10 = 90%**

| Query | Expected section | Top-1 hit | Top-3 hit | Top score |
|---|---|---|---|---|
| how much is the registration fee | Registration Fee | yes | yes | 0.915 |
| ስንት ብር ነው ክፍያው? | Registration Fee | yes | yes | 0.814 |
| what documents do I need | Registration Requirements | yes | yes | 0.870 |
| ምን አይነት ወረቀት ያስፈልጋል? | Registration Requirements | yes | yes | 0.833 |
| when does registration open | Registration Dates | no | yes | 0.911 |
| registration meche yikefetal? | Registration Dates | no | no | 0.827 |
| where is the office located | Where to Register | yes | yes | 0.841 |
| registration office yet new? | Where to Register | yes | yes | 0.867 |
| my ID card is lost | Common Problems | yes | yes | 0.886 |
| I have a complaint who do I talk to | Contact | yes | yes | 0.830 |
