# <div align="center">Predictively Combatting Toxicity in Health-related Online Discussions through Machine Learning</div>

### <div align="center"> Anonymous authors (for now!)</div>

##### <div align="center"> <i>Under review on <b>IJCNN 2025</b></i></div>

<h3>Abstract</h3>

In health-related topics, user toxicity in online discussions frequently becomes a source of social conflict or promotion of dangerous, unscientific behaviour; common approaches for battling it include different forms of detection, flagging and/or removal of existing toxic comments, which is often counterproductive for platforms and users alike. In this work, we propose the alternative of combatting user toxicity predictively, anticipating where a user could interact toxically in health-related online discussions. Applying a Collaborative Filtering-based Machine Learning methodology, we predict the toxicity in COVID-related conversations between any user and subcommunity of Reddit, surpassing 80% predictive performance in relevant metrics, and allowing us to prevent the pairing of conflicting users and subcommunities.

<h3>Data</h3>

The dataset used in this work was obtained using PushShift.io API, and consists of 1.5 million Reddit comments discussing COVID-19. These comments were posted between January and June of 2021 and were obtained during February of 2023. To preserve anonymicity, and due to the large dataset size, we currently only provide a [temporary download link](https://udcgal-my.sharepoint.com/:x:/g/personal/j_ruza_udc_es/EV-UQDFFPulGntuaGjVXu94BRQ9AP9WUVElA7DPoKFL4Yw?e=XPqnOP) which will expire on 26/01/2026, and will be posted on Zenodo upon acceptance.

<h3>Reproducibility</h3>

The provided framework was developed in Python 3.9 and uses CUDA 11.1 for GPU acceleration. 
If set up correctly, `python3 main.py` should run the full pipeline, from data preprocessing to model evaluation. 


<h3>Licensing</h3>

```
MIT License

Copyright (c) 2025 Anonymous authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```