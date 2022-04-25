# EyeGazeSurvey
Automatic Gaze Analysis: A Survey of Deep Learning based Approaches by Shreya Ghosh, Abhinav Dhall, Munawar Hayat, Jarrod Knibbe and Qiang Ji. [(Paper Link)](https://arxiv.org/pdf/2108.05479.pdf)
  
If we miss your work, please let us know and we'll add it. 

## Contact
- <a href="https://sites.google.com/view/shreyaghosh/home">Shreya Ghosh</a>.

<!---## Update-->

If you find the survey useful for your research, please consider citing our work:
```
@article{ghosh2021Automatic,
  title={Automatic Gaze Analysis: A Survey of Deep Learning based Approaches},
  author={Ghosh, Shreya and Dhall, Abhinav and Hayat, Munawar and Knibbe, Jarrod and Ji, Qiang},
  journal={arXiv preprint arXiv:2108.05479},
  year={2021}
}
```


## Datasets
A comparison of gaze datasets with respect to several attributes (i.e. number of subjects (\# subjects), gaze labels, modality, headpose and gaze angle in yaw and pitch axis, environment (Env.), baseline method, data statistics (\# of data), and year of publication.) The abbreviations used are: In: Indoor, Out: Outdoor, Both: Indoor + Outdoor, Gen.: Generation, u/k: unknown, Seq.: Sequence, VF: Visual Field, EB: Eye Blink, GE: Gaze Event, GBRT: Gradient Boosting Regression Trees, GC: Gaze Communication, GNN: Graph Neural Network and Seg.: Segmentation.
![datasets](/images/datasets.png)


## Gaze Analysis Methods
A comparison of gaze analysis methods with respect to registration (Reg.), representation (Represent.), Level of Supervision, Model, Prediction, validation on benchmark datasets (validation), Platforms, Publication venue (Publ.) and year. Here, GV: Gaze Vector, Scr.: Screen, LOSO: Leave One Subject Out, LPIPS: Learned Perceptual Image Patch Similarity, MM: Morphable Model, RRF: Random Regression Forest, AEM: Anatomic Eye Model, GRN: Gaze Regression Network, ET: External Target, FV: Free Viewing, HH: HandHeld Device, HMD: Head Mounted Device, Seg.: Segmentation and GR: Gaze Redirection, LAEO: Looking At Each Other.
![datasets](/images/prior_work.png)

# Eye-Gaze: [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com) 
A curated list of papers and datsets for various gaze estimation techniques, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision). Mostly Recent papers are here.

## Contents
 - [Eye Gaze Estimation/Tracking](#Eye-Gaze-Estimation)
 - [Gaze Trajectory](#Gaze-Trajectory)
 - [Gaze Redirection](#Gaze-Redirection)
 - [Gaze Zone + Driver Gaze](#Gaze-Zone)
 - [Gaze and Attention](#Gaze-and-Attention)
 - [Gaze and Interaction](#Gaze-and-Interaction)
 - [Visual Attention](#Visual-Attention)
 - [Uncategorized Papers](#Uncategorized-Papers)
<!--  - [Looking At Each Other](#LAEO)  -->
<!---- [Cross-modal Headpose (Audio-Video / Video-Audio)](#Cross-modal-Generation-(Audio-Video--Video-Audio)) [Multi-modal Architectures](#Multi-modal-Architectures)-->

#### Eye Gaze Estimation/Tracking 
* ESCNet: Gaze Target Detection with the Understanding of 3D Scenes - Jun Bao, Buyu Liu, Jun Yu CVPR2022 [pdf NA]
* [Dynamic 3D Gaze from Afar: Deep Gaze Estimation from Temporal Eye-Head-Body Coordination](https://vision.ist.i.kyoto-u.ac.jp/pubs/SNonaka_CVPR22.pdf) - Soma Nonaka, Shohei Nobuhara, Ko Nishino CVPR2022
* [GaTector: A Unified Framework for Gaze Object Prediction](https://arxiv.org/pdf/2112.03549.pdf) - Binglu Wang, Tao Hu, Baoshan Li, Xiaojuan Chen, Zhijie Zhang CVPR2022
* [GazeOnce: Real-Time Multi-Person Gaze Estimation](https://arxiv.org/abs/2204.09480) - Mingfang Zhang, Yunfei Liu, Feng Lu CVPR2022
* [End-to-End Human-Gaze-Target Detection with Transformers](https://arxiv.org/abs/2203.10433) - Danyang Tu, Xiongkuo Min, Huiyu Duan, Guodong Guo, Guangtao Zhai, Wei Shen CVPR2022
* [Cross-Encoder for Unsupervised Gaze Representation Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Cross-Encoder_for_Unsupervised_Gaze_Representation_Learning_ICCV_2021_paper.pdf) - Yunjia Sun, Jiabei Zeng, Shiguang Shan, Xilin Chen ICCV2021
* [Vulnerability of Appearance-based Gaze Estimation](https://arxiv.org/pdf/2103.13134.pdf)
* [Generalizing Gaze Estimation with Outlier-guided Collaborative Adaptation](https://arxiv.org/pdf/2107.13780.pdf)
* [Weakly-Supervised Physically Unconstrained Gaze Estimation](https://openaccess.thecvf.com/content/CVPR2021/papers/Kothari_Weakly-Supervised_Physically_Unconstrained_Gaze_Estimation_CVPR_2021_paper.pdf) - Rakshit Kothari, Shalini De Mello, Umar Iqbal, Wonmin Byeon, Seonwook Park, Jan Kautz (CVPR 2021) 
* [Goal-Oriented Gaze Estimation for Zero-Shot Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Goal-Oriented_Gaze_Estimation_for_Zero-Shot_Learning_CVPR_2021_paper.pdf) - Yang Liu, Lei Zhou, Xiao Bai, Yifei Huang, Lin Gu, Jun Zhou, Tatsuya Harada  (CVPR 2021)
* [GOO: A Dataset for Gaze Object Prediction in Retail Environments](https://arxiv.org/pdf/2105.10793) - Henri Tomas, Marcus Reyes, Raimarc Dionido, Mark Ty, Jonric Mirando, Joel Casimiro, Rowel Atienza, Richard Guinto  (CVPRW 2021)
* [PureGaze: Purifying Gaze Feature for Generalizable Gaze Estimation](https://arxiv.org/pdf/2103.13173)  - Yihua Cheng, Yiwei Bao, Feng Lu  
* [Gaze Estimation with an Ensemble of Four Architectures](https://arxiv.org/pdf/2107.01980) - Xin Cai, Boyu Chen, Jiabei Zeng, Jiajun Zhang, Yunjia Sun, Xiao Wang, Zhilong Ji, Xiao Liu, Xilin Chen, Shiguang Shan  
* [The Story in Your Eyes: An Individual-difference-aware Model for Cross-person Gaze Estimation](https://arxiv.org/pdf/2107.01980) - Jun Bao, Buyu Liu, Jun Yu  
* [Bayesian Eye Tracking](https://arxiv.org/pdf/2106.13387) - Qiang Ji, Kang Wang  
* [Glance-and-Gaze Vision Transformer](https://arxiv.org/pdf/2106.02277) -	Qihang Yu, Yingda Xia, Yutong Bai, Yongyi Lu, Alan Yuille, Wei Shen  
* [Gaze Estimation using Transformer](https://arxiv.org/pdf/2105.14424)  - 	Yihua Cheng, Feng Lu
* [Self-supervised learning through the eyes of a child.](https://papers.nips.cc/paper/2020/file/7183145a2a3e0ce2b68cd3735186b1d5-Paper.pdf)  
  Emin Orhan, Vaibhav Gupta, Brenden M. Lake  
* [A Coarse-to-Fine Adaptive Network for Appearance-Based Gaze Estimation.](https://aaai.org/ojs/index.php/AAAI/article/view/6636/6490)  
  Yihua Cheng, Shiyao Huang, Fei Wang, Chen Qian, Feng Lu  
* [ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500358.pdf) -  Xucong Zhang, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang , Otmar Hilliges   
* [Towards End-to-end Video-based Eye-Tracking.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570732.pdf)  -   Seonwook Park, Emre Aksan, Xucong Zhang, Otmar Hilliges
* [Deep Learning-based Pupil Center Detection for Fast and Accurate Eye Tracking System.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640035.pdf) -  Kang Il Lee, Jung Ho Jeon, Byung Cheol Song   
* [Unsupervised Representation Learning for Gaze Estimation.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Unsupervised_Representation_Learning_for_Gaze_Estimation_CVPR_2020_paper.pdf) -  Yu Yu, Jean-Marc Odobez  
* [Domain Adaptation Gaze Estimation by Embedding with Prediction Consistency.](https://openaccess.thecvf.com/content/ACCV2020/papers/Guo_Domain_Adaptation_Gaze_Estimation_by_Embedding_with_Prediction_Consistency_ACCV_2020_paper.pdf) -   Zidong Guo, Zejian Yuan, Chong Zhang, Wanchao Chi, Yonggen Ling, Shenghao Zhang   
* [Offset Calibration for Appearance-Based Gaze Estimation via Gaze Decomposition.](https://openaccess.thecvf.com/content_WACV_2020/papers/Chen_Offset_Calibration_for_Appearance-Based_Gaze_Estimation_via_Gaze_Decomposition_WACV_2020_paper.pdf)  -   Zhaokang Chen, Bertram Shi   
* [Gaze Estimation for Assisted Living Environments.](https://openaccess.thecvf.com/content_WACV_2020/papers/Dias_Gaze_Estimation_for_Assisted_Living_Environments_WACV_2020_paper.pdf)  -   Philipe Ambrozio Dias, Damiano Malafronte, Henry Medeiros, Francesca Odone   
* [Learning to Detect Head Movement in Unconstrained Remote Gaze Estimation in the Wild.](https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_Learning_to_Detect_Head_Movement_in_Unconstrained_Remote_Gaze_Estimation_WACV_2020_paper.pdf)  -   Zhecan Wang, Jian Zhao, Cheng Lu, Fan Yang, Han Huang, lianji li, Yandong Guo  
* [Learning-based Region Selection for End-to-End Gaze Estimation.](https://www.bmvc2020-conference.com/assets/papers/0086.pdf) -   Xucong Zhang, Yusuke Sugano, Andreas Bulling and Otmar Hilliges  
* [Low Cost Gaze Estimation: Knowledge-Based Solutions.](https://doi.org/10.1109/TIP.2019.2946452) - Ion Martinikorena, Andoni Larumbe-Bergera, Mikel Ariz, Sonia Porta, Rafael Cabeza, Arantxa Villanueva (2020 IEEE TIP)
* [Gaze Estimation by Exploring Two-Eye Asymmetry.](https://doi.org/10.1109/TIP.2020.2982828)- Yihua Cheng, Xucong Zhang, Feng Lu, Yoichi Sato (2020 IEEE TIP)
* [Gaze360: Physically Unconstrained Gaze Estimation in the Wild.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.pdf) - Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Matusik, Antonio Torralba  
* [Few-Shot Adaptive Gaze Estimation.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf)  
  Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges, Jan Kautz   
* [Mixed Effects Neural Networks (MeNets) With Applications to Gaze Estimation.]
  Yunyang Xiong, Hyunwoo J. Kim, Vikas Singh  
* [Generalizing Eye Tracking With Bayesian Adversarial Learning.](https://www.semanticscholar.org/paper/Generalizing-Eye-Tracking-with-Bayesian-Adversarial-Wang-Zhao/77b9b6786699a236aad0c3fa3734730ece4a780f)  Kang Wang, Rui Zhao, Hui Su, Qiang Ji  
* [Eyemotion: Classifying Facial Expressions in VR Using Eye-Tracking Cameras.](https://doi.org/10.1109/WACV.2019.00178)  
  Steven Hickson, Nick Dufour, Avneesh Sud, Vivek Kwatra, Irfan A. Essa  
* [MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation.](https://ieeexplore.ieee.org/document/8122058) - Xucong Zhang and Yusuke Sugano and Mario Fritz and Andreas Bulling (2017 IEEE TPAMI) 
* [InvisibleEye: Mobile Eye Tracking Using Multiple Low-Resolution Cameras and Learning-Based Gaze Estimation.](https://dl.acm.org/citation.cfm?id=3130971) -  Marc Tonsen and Julian Steil and Yusuke Sugano and Andreas Bulling 
* [A low-cost and calibration-free gaze estimator for soft biometrics: An explorative study.](https://www.sciencedirect.com/science/article/pii/S0167865515003669)  - Dario Cazzato and Andrea Evangelista and Marco Leo and Pierluigi Carcagní and Cosimo Distante
* [Model-based head pose-free gaze estimation for assistive communication.](https://www.sciencedirect.com/science/article/pii/S1077314216000667) - Stefania Cristina and Kenneth P. Camilleri  
* [Fast and accurate algorithm for eye localisation for gaze tracking in low-resolution images.](http://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2015.0316)  
  Anjith George and Aurobinda Routray  
* [Gaze Estimation in the 3D Space Using RGB-D Sensors.](http://publications.idiap.ch/index.php/publications/show/3228) - Kenneth A. Funes-Mora and Jean-Marc Odobez  

#### Gaze Trajectory
* [Appearance-Based Gaze Estimation via Uncalibrated Gaze Pattern Recovery.](https://ieeexplore.ieee.org/document/7833091) - Feng Lu and Xiaowu Chen and Yoichi Sato
* [Dynamic Fusion of Eye Movement Data and Verbal Narrations in Knowledge-rich Domains.](https://papers.nips.cc/paper/2020/file/16837163fee34175358a47e0b51485ff-Paper.pdf)  
  Ervine Zheng, Qi Yu, Rui Li, Pengcheng Shi, Anne Haake 
* [Neuro-Inspired Eye Tracking With Eye Movement Dynamics.](http://homepages.rpi.edu/~wangk10/papers/wang2019neural.pdf) -  Kang Wang, Hui Su, Qiang Ji

#### Gaze Redirection
* [CUDA-GR: Controllable Unsupervised Domain Adaptation for Gaze Redirection](https://arxiv.org/pdf/2106.10852) - Swati Jindal, Xin Eric Wang   
* [Self-Learning Transformations for Improving Gaze and Head Redirection.](https://papers.nips.cc/paper/2020/file/98f2d76d4d9caf408180b5abfa83ae87-Paper.pdf)  
  Yufeng Zheng, Seonwook Park, Xucong Zhang, Shalini De Mello, Otmar Hilliges 
* [Photo-Realistic Monocular Gaze Redirection Using Generative Adversarial Networks.](http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Photo-Realistic_Monocular_Gaze_Redirection_Using_Generative_Adversarial_Networks_ICCV_2019_paper.pdf)  - Zhe He, Adrian Spurr, Xucong Zhang, Otmar Hilliges 
* [Improving Few-Shot User-Specific Gaze Adaptation via Gaze Redirection Synthesis.](http://www.idiap.ch/~odobez/publications/YuLiuOdobez-CVPR2019.pdf)  
  Yu Yu, Gang Liu, Jean-Marc Odobez 
#### Gaze Zone + Driver Gaze
* [Driver Gaze Region Estimation Without Using Eye Movement](https://ieeexplore.ieee.org/document/7478592)  - Lex Fridman and Philipp Langhans and Joonbum Lee and Bryan Reimer
* [EyeGAN: Gaze-Preserving, Mask-Mediated Eye Image Synthesis.](https://openaccess.thecvf.com/content_WACV_2020/papers/Kaur_EyeGAN_Gaze-Preserving_Mask-Mediated_Eye_Image_Synthesis_WACV_2020_paper.pdf)  -  Harsimran Kaur, Roberto Manduchi 
* ["Looking at the Right Stuff" - Guided Semantic-Gaze for Autonomous Driving.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pal_Looking_at_the_Right_Stuff_-_Guided_Semantic-Gaze_for_Autonomous_CVPR_2020_paper.pdf) -   Anwesan Pal, Sayan Mondal, Henrik I. Christensen   

<!-- 
#### Looking At Each Other
* Detecting people looking at each other in videos (link)
LAEO-Net: revisiting people Looking At Each Other in videos (link)
LAEO-Net++: revisiting people Looking At Each Other in videos (link) -->

#### Gaze and Attention
* [Dual Attention Guided Gaze Target Detection in the Wild](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_Dual_Attention_Guided_Gaze_Target_Detection_in_the_Wild_CVPR_2021_paper.pdf) - Yi Fang, Jiapeng Tang, Wang Shen, Wei Shen, Xiao Gu, Li Song, Guangtao Zhai 
* [Connecting What To Say With Where To Look by Modeling Human Attention Traces](https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_Connecting_What_To_Say_With_Where_To_Look_by_Modeling_CVPR_2021_paper.pdf) - Zihang Meng, Licheng Yu, Ning Zhang, Tamara L. Berg, Babak Damavandi, Vikas Singh, Amy Bearman  
* [Appearance-Based Gaze Estimation Using Attention and Difference Mechanism](https://openaccess.thecvf.com/content/CVPR2021W/GAZE/papers/D_Appearance-Based_Gaze_Estimation_Using_Attention_and_Difference_Mechanism_CVPRW_2021_paper.pdf) - Murthy L R D, Pradipta Biswas  
* [Visual Focus of Attention Estimation in 3D Scene With an Arbitrary Number of Targets](https://openaccess.thecvf.com/content/CVPR2021W/GAZE/papers/Siegfried_Visual_Focus_of_Attention_Estimation_in_3D_Scene_With_an_CVPRW_2021_paper.pdf) - Remy Siegfried, Jean-Marc Odobez  
* [Augmented saliency model using automatic 3D head pose detection and learned gaze following in natural scenes.](https://www.sciencedirect.com/science/article/pii/S0042698914002739) - Daniel Parks and Ali Borji and Laurent Itti 
* [Improving Natural Language Processing Tasks with Human Gaze-Guided Neural Attention.](https://papers.nips.cc/paper/2020/file/460191c72f67e90150a093b4585e7eb4-Paper.pdf)  
  Ekta Sood, Simon Tannert, Philipp Mueller, Andreas Bulling 

#### Gaze and Interaction
* [Glance and Gaze: Inferring Action-Aware Points for One-Stage Human-Object Interaction Detection](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_Glance_and_Gaze_Inferring_Action-Aware_Points_for_One-Stage_Human-Object_Interaction_CVPR_2021_paper.pdf) - Xubin Zhong, Xian Qu, Changxing Ding, Dacheng Tao
* [Appearance-Based Gaze Estimation With Online Calibration From Mouse Operations.](https://ieeexplore.ieee.org/document/7050250)  Yusuke Sugano and Yasuyuki Matsushita and Yoichi Sato and Hideki Koike  
  

#### Visual Attention
* [Where are they looking?](https://papers.nips.cc/paper/2015/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html) - Adria Recasens, Aditya Khosla, Carl Vondrick, Antonio Torralba. NIPS 2015
* [Connecting Gaze, Scene, and Attention: Generalized Attention Estimation via Joint Modeling of Gaze and Scene Saliency](https://www.springerprofessional.de/connecting-gaze-scene-and-attention-generalized-attention-estima/16179862) - Eunji Chong, Nataniel Ruiz, Yongxin Wang, Yun Zhang, Agata Rozga, James M. Rehg. ECCV2018 
* [Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning](http://openaccess.thecvf.com/content_ICCV_2019/papers/Fan_Understanding_Human_Gaze_Communication_by_Spatio-Temporal_Graph_Reasoning_ICCV_2019_paper.pdf) -   Lifeng Fan, Wenguan Wang, Siyuan Huang, Xinyu Tang, Song-Chun Zhu. ICCV2019
* [Detecting Attended Visual Targets in Video](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chong_Detecting_Attended_Visual_Targets_in_Video_CVPR_2020_paper.pdf) - Eunji Chong, Yongxin Wang, Nataniel Ruiz, James M. Rehg. CVPR2020


#### Uncategorized Papers
* [PupilTAN: A Few-Shot Adversarial Pupil Localizer](https://openaccess.thecvf.com/content/CVPR2021W/GAZE/papers/Poulopoulos_PupilTAN_A_Few-Shot_Adversarial_Pupil_Localizer_CVPRW_2021_paper.pdf) - Nikolaos Poulopoulos, Emmanouil Z. Psarakis, Dimitrios Kosmopoulos 
* [How is Gaze Influenced by Image Transformations? Dataset and Model.](https://doi.org/10.1109/TIP.2019.2945857) - Zhaohui Che, Ali Borji, Guangtao Zhai, Xiongkuo Min, Guodong Guo, Patrick Le Callet (2020 IEEE TIP) [[code]](https://github.com/CZHQuality/Sal-CFS-GAN)
* [When Computer Vision Gazes at Cognition.](https://arxiv.org/abs/1412.2672) - Tao Gao and Daniel Harari and Joshua Tenenbaum and Shimon Ullman
* [A Novel Approach to Real-time Non-intrusive Gaze Finding.](http://www.bmva.org/bmvc/1998/papers/d058/h058.htm) - L.-Q. Xu and D. Machin and P. Sheppard  
* [Non-Intrusive Gaze Tracking Using Artificial Neural Networks.](https://papers.nips.cc/paper/863-non-intrusive-gaze-tracking-using-artificial-neural-networks) - Shumeet Baluja and Dean Pomerleau   
* [Interact as You Intend: Intention-Driven Human-Object Interaction Detection.](https://arxiv.org/abs/1808.09796) - Bingjie Xu and Junnan Li and Yongkang Wong and Mohan S. Kankanhalli and Qi Zhao   
* [TurkerGaze: Crowdsourcing Saliency with Webcam based Eye Tracking.](https://arxiv.org/abs/1504.06755) - Pingmei Xu and Krista A Ehinger and Yinda Zhang and Adam Finkelstein and Sanjeev R. Kulkarni and Jianxiong Xiao 
  

<!----
  
* [Gaze Tracking System for User Wearing Glasses.](https://www.mdpi.com/1424-8220/14/2/2110) -  Su Gwon and Chul Cho and Hyeon Lee and Won Lee and Kang Park  
* [Adaptive Linear Regression for Appearance-Based Gaze Estimation.](https://ieeexplore.ieee.org/document/6777326)   Feng Lu and Yusuke Sugano and Takahiro Okabe and Yoichi Sato  
* [Learning gaze biases with head motion for head pose-free gaze estimation.](https://www.sciencedirect.com/science/article/pii/S0262885614000171)  
  Feng Lu and Takahiro Okabe and Yusuke Sugano and Yoichi Sato  
* [Real-Time Gaze Estimation with Online Calibration.](https://ieeexplore.ieee.org/document/6916495)  
  Li Sun and Mingli Song and Zicheng Liu and Ming-Ting Sun  
* [Appearance-Based Gaze Estimation Using Visual Saliency.](https://ieeexplore.ieee.org/document/6193107)  
  Yusuke Sugano and Yasuyuki Matsushita and Yoichi Sato  
* [Robust Eye and Pupil Detection Method for Gaze Tracking.](journals.sagepub.com/doi/full/10.5772/55520)  
  Su Yeong Gwon and Chul Woo Cho and Hyeon Chang Lee and Won Oh Lee and Kang Ryoung Park  
* [Real-time eye gaze tracking for gaming design and consumer electronics systems.](https://ieeexplore.ieee.org/document/6227433)  
  Peter Corcoran and Florin Nanu and Stefan Petrescu and Petronel Bigioi  
* [Combining Head Pose and Eye Location Information for Gaze Estimation](https://ieeexplore.ieee.org/document/5959981)  
  R. Valenti and N. Sebe and T. Gevers  
* [Gaze tracking system at a distance for controlling IPTV.](https://ieeexplore.ieee.org/document/5681143)  
  Hyeon Lee and Duc Luong and Chul Cho and Eui Lee and Kang Park  

* [Identifying Children with Autism Spectrum Disorder Based on Gaze-Following.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9190831)  
  	Yi Fang, Huiyu Duan, Fangyu Shi, Xiongkuo Min, Guangtao Zhai  



* [Lian-etal2019 AAAI] RGBD Based Gaze Estimation via Multi-Task CNN.  (https://doi.org/10.1609/aaai.v33i01.33012488) - Dongze Lian, Ziheng Zhang, Weixin Luo, Lina Hu, Minye Wu, Zechao Li, Jingyi Yu, Shenghua Gao  
* [Fischer-etal2018] RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments.   (https://link.springer.com/chapter/10.1007%2F978-3-030-01249-6_21  )  
  Tobias Fischer, Hyung Jin Chang, Yiannis Demiris  

   * [Yohanandan-etal2018] Saliency Preservation in Low-Resolution Grayscale Images.   (https://link.springer.com/chapter/10.1007/978-3-030-01231-1_15)  
  Shivanthan Yohanandan, Andy Song, Adrian G. Dyer, Dacheng Tao  

   * [Jiang-etal2018] DeepVS: A Deep Learning Based Video Saliency Prediction Approach.   (https://link.springer.com/chapter/10.1007/978-3-030-01264-9_37)  
  Lai Jiang, Mai Xu, Tie Liu, Minglang Qiao, Zulin Wang  

   * [Kummerer-etal2018] Saliency Benchmarking Made Easy: Separating Models, Maps and Metrics.   (https://link.springer.com/chapter/10.1007/978-3-030-01270-0_47)  
  Matthias Kummerer, Thomas S. A. Wallis, Matthias Bethge  

   * [Zheng-etal2018] Task-driven Webpage Saliency.   (https://link.springer.com/chapter/10.1007/978-3-030-01264-9_18)  
  Quanlong Zheng, Jianbo Jiao, Ying Cao, Rynson W.H. Lau  

   * [Zhang-etal2018] Saliency Detection in 360° Videos.   (https://link.springer.com/chapter/10.1007/978-3-030-01234-2_30)  
  Ziheng Zhang, Yanyu Xu, Jingyi Yu, Shenghua Gao  

   * [Song-etal2018] Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection.   (https://link.springer.com/chapter/10.1007/978-3-030-01252-6_44  )  
  Hongmei Song, Wenguan Wang, Sanyuan Zhao, Jianbing Shen, Kin-Man Lam  

   * [Fan-etal2018] Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground.   (https://link.springer.com/chapter/10.1007/978-3-030-01267-0_12  )  
  Deng-Ping Fan, Ming-Ming Cheng, Jiang-Jiang Liu, Shang-Hua Gao, Qibin Hou, Ali Borji  

   * [Li-etal2018] Contour Knowledge Transfer for Salient Object Detection.   (https://link.springer.com/chapter/10.1007/978-3-030-01267-0_22  )  
  Xin Li, Fan Yang, Hong Cheng, Wei Liu, Dinggang Shen  

   * [Fan-etal2018] Associating Inter-image Salient Instances for Weakly Supervised Semantic Segmentation.   (https://link.springer.com/chapter/10.1007/978-3-030-01240-3_23  )  
  Ruochen Fan, Qibin Hou, Ming-Ming Cheng, Gang Yu, Ralph R. Martin, Shi-Min Hu  

   * [Chen-etal2018] Reverse Attention for Salient Object Detection.   (https://link.springer.com/chapter/10.1007/978-3-030-01240-3_15  )  
  Shuhan Chen, Xiuli Tan, Ben Wang, Xuelong Hu  


   * [Fan-etal2019] Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Fan_Understanding_Human_Gaze_Communication_by_Spatio-Temporal_Graph_Reasoning_ICCV_2019_paper.pdf)  
  Lifeng Fan, Wenguan Wang, Siyuan Huang, Xinyu Tang, Song-Chun Zhu   

   * [Kellnhofer-etal2019] Gaze360: Physically Unconstrained Gaze Estimation in the Wild. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.pdf)  
  Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Matusik, Antonio Torralba   

   * [He-etal2019] Photo-Realistic Monocular Gaze Redirection Using Generative Adversarial Networks. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Photo-Realistic_Monocular_Gaze_Redirection_Using_Generative_Adversarial_Networks_ICCV_2019_paper.pdf)  
  Zhe He, Adrian Spurr, Xucong Zhang, Otmar Hilliges   

   * [Park-etal2019] Few-Shot Adaptive Gaze Estimation. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf)  
  Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges, Jan Kautz   

   * [Xiong-etal2019] Mixed Effects Neural Networks (MeNets) With Applications to Gaze Estimation. <a href=")  
  Yunyang Xiong, Hyunwoo J. Kim, Vikas Singh  

   * [Yu-etal2019] Improving Few-Shot User-Specific Gaze Adaptation via Gaze Redirection Synthesis. <a href="http://www.idiap.ch/~odobez/publications/YuLiuOdobez-CVPR2019.pdf)  
  Yu Yu, Gang Liu, Jean-Marc Odobez  

   * [Wang-etal2019] Neuro-Inspired Eye Tracking With Eye Movement Dynamics. <a href="http://homepages.rpi.edu/~wangk10/papers/wang2019neural.pdf)  
  Kang Wang, Hui Su, Qiang Ji  

   * [Wang-etal2019] Generalizing Eye Tracking With Bayesian Adversarial Learning. <a href="https://www.semanticscholar.org/paper/Generalizing-Eye-Tracking-with-Bayesian-Adversarial-Wang-Zhao/77b9b6786699a236aad0c3fa3734730ece4a780f)  
  Kang Wang, Rui Zhao, Hui Su, Qiang Ji  

  * [Hickson-etal2019 WACV] Eyemotion: Classifying Facial Expressions in VR Using Eye-Tracking Cameras. <a href="https://doi.org/10.1109/WACV.2019.00178)  
  Steven Hickson, Nick Dufour, Avneesh Sud, Vivek Kwatra, Irfan A. Essa  

   * [Lian-etal2019 AAAI] RGBD Based Gaze Estimation via Multi-Task CNN.<a href="https://doi.org/10.1609/aaai.v33i01.33012488)  
  Dongze Lian, Ziheng Zhang, Weixin Luo, Lina Hu, Minye Wu, Zechao Li, Jingyi Yu, Shenghua Gao  
   * [Fischer-etal2018] RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. <a href="https://link.springer.com/chapter/10.1007%2F978-3-030-01249-6_21" rel="nofollow)  
  Tobias Fischer, Hyung Jin Chang, Yiannis Demiris  

   * [Yohanandan-etal2018] Saliency Preservation in Low-Resolution Grayscale Images. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01231-1_15)  
  Shivanthan Yohanandan, Andy Song, Adrian G. Dyer, Dacheng Tao  

   * [Jiang-etal2018] DeepVS: A Deep Learning Based Video Saliency Prediction Approach. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01264-9_37)  
  Lai Jiang, Mai Xu, Tie Liu, Minglang Qiao, Zulin Wang  

   * [Kummerer-etal2018] Saliency Benchmarking Made Easy: Separating Models, Maps and Metrics. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01270-0_47)  
  Matthias Kummerer, Thomas S. A. Wallis, Matthias Bethge  

   * [Zheng-etal2018] Task-driven Webpage Saliency. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01264-9_18)  
  Quanlong Zheng, Jianbo Jiao, Ying Cao, Rynson W.H. Lau  

   * [Zhang-etal2018] Saliency Detection in 360° Videos. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01234-2_30)  
  Ziheng Zhang, Yanyu Xu, Jingyi Yu, Shenghua Gao  

   * [Song-etal2018] Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01252-6_44" rel="nofollow)  
  Hongmei Song, Wenguan Wang, Sanyuan Zhao, Jianbing Shen, Kin-Man Lam  

   * [Fan-etal2018] Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01267-0_12" rel="nofollow)  
  Deng-Ping Fan, Ming-Ming Cheng, Jiang-Jiang Liu, Shang-Hua Gao, Qibin Hou, Ali Borji  

   * [Li-etal2018] Contour Knowledge Transfer for Salient Object Detection. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01267-0_22" rel="nofollow)  
  Xin Li, Fan Yang, Hong Cheng, Wei Liu, Dinggang Shen  

   * [Fan-etal2018] Associating Inter-image Salient Instances for Weakly Supervised Semantic Segmentation. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01240-3_23" rel="nofollow)  
  Ruochen Fan, Qibin Hou, Ming-Ming Cheng, Gang Yu, Ralph R. Martin, Shi-Min Hu  

   * [Chen-etal2018] Reverse Attention for Salient Object Detection. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01240-3_15" rel="nofollow)  
  Shuhan Chen, Xiuli Tan, Ben Wang, Xuelong Hu  

      * [Brau-etal2018] Multiple-Gaze Geometry: Inferring Novel 3D Locations from Gazes Observed in Monocular Video.    (https://link.springer.com/chapter/10.1007/978-3-030-01225-0_38)
   Ernesto Brau, Jinyan Guan, Tanya Jeffries, Kobus Barnard   

      * [Recasens-etal2018] Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks.    (https://link.springer.com/chapter/10.1007/978-3-030-01240-3_4)
   Adrià Recasens, Petr Kellnhofer, Simon Stent, Wojciech Matusik, Antonio Torralba   

      * [Li-etal2018] In the Eye of Beholder: Joint Learning of Gaze and Actions in First Person Video.    (https://link.springer.com/chapter/10.1007/978-3-030-01228-1_38)
   Yin Li, Miao Liu, James M. Rehg   

      * [Huang-etal2018] Predicting Gaze in Egocentric Video by Learning Task-Dependent Attention Transition.    (https://link.springer.com/chapter/10.1007/978-3-030-01225-0_46)
   Yifei Huang, Minjie Cai, Zhenqiang Li, Yoichi Sato   

      * [Produkin-etal2018] Deep Directional Statistics: Pose Estimation with Uncertainty Quantification.    (https://link.springer.com/chapter/10.1007/978-3-030-01240-3_33)
   Sergey Prokudin, Peter Gehler , Sebastian Nowozin   

      * [Chong-etal2018] Connecting Gaze, Scene, and Attention: Generalized Attention Estimation via Joint Modeling of Gaze and Scene Saliency.    (https://link.springer.com/chapter/10.1007%2F978-3-030-01228-1_24)
   Eunji Chong and Nataniel Ruiz and Yongxin Wang and Yun Zhang and Agata Rozga and James Rehg   

      * [Park-etal2018] Deep Pictorial Gaze Estimation.    (https://arxiv.org/abs/1807.10002)
   Seonwook Park and Adrian Spurr and Otmar Hilliges   

      * [Cheng-etal2018] Appearance-Based Gaze Estimation via Evaluation-Guided Asymmetric Regression.    (http://openaccess.thecvf.com/content_ECCV_2018/html/Yihua_Cheng_Appearance-Based_Gaze_Estimation_ECCV_2018_paper.html)
   Cheng, Yihua and Lu, Feng and Zhang, Xucong   

      * [Yu-etal2018 ECCVW] Deep Multitask Gaze Estimation with a Constrained Landmark-Gaze Model.    (https://link.springer.com/chapter/10.1007/978-3-030-11012-3_35)
   Yu Yu, Gang Liu, Jean-Marc Odobez   

      * [Zhang-etal2018] Deep Unsupervised Saliency Detection: A Multiple Noisy Labeling Perspective.    (https://ieeexplore.ieee.org/document/8579039)</a>
   Jing Zhang, Tong Zhang, Yuchao Dai, Mehrtash Harandi, Richard Hartley   

      * [Yu-etal2018] Recurrent Saliency Transformation Network: Incorporating Multi-Stage Visual Cues for Small Organ Segmentation.    (https://ieeexplore.ieee.org/document/8578962)</a>
   Qihang Yu, Lingxi Xie, Yan Wang, Yuyin Zhou, Elliot K. Fishman, Alan L. Yuille   

      * [Gorji-etal2018] Going From Image to Video Saliency: Augmenting Image Salience With Dynamic Attentional Push.    (https://ieeexplore.ieee.org/document/8578881)</a>
   Siavash Gorji, James J. Clark   

      * [Islam-etal2018] Revisiting Salient Object Detection: Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects.    (https://ieeexplore.ieee.org/document/8578844)</a>
   Md Amirul Islam, Mahmoud Kalash, Neil D. B. Bruce   

      * [Wang-etal2018] Revisiting Video Saliency: A Large-Scale Benchmark and a New Model.    (https://ieeexplore.ieee.org/document/8578612)</a>
   Wenguan Wang, Jianbing Shen, Fang Guo, Ming-Ming Cheng, Ali Borji   

      * [Li-etal2018] Flow Guided Recurrent Neural Encoder for Video Salient Object Detection.    (https://ieeexplore.ieee.org/document/8578440)</a>
   Guanbin Li, Yuan Xie, Tianhao Wei, Keze Wang, Liang Lin   

      * [Wang-etal2018] Detect Globally, Refine Locally: A Novel Approach to Saliency Detection.    (https://ieeexplore.ieee.org/document/8578428)</a>
   Tiantian Wang, Lihe Zhang, Shuo Wang, Huchuan Lu, Gang Yang, Xiang Ruan, Ali Borji   

      * [Liu-etal2018] PiCANet: Learning Pixel-Wise Contextual Attention for Saliency Detection.    (https://ieeexplore.ieee.org/document/8578424)</a>
   Nian Liu, Junwei Han, Ming-Hsuan Yang   

      * [Chen-Li2018] Progressively Complementarity-Aware Fusion Network for RGB-D Salient Object Detection.    (https://ieeexplore.ieee.org/document/8578420)</a>
   Hao Chen, Youfu Li   

      * [Wang-etal2018] Salience Guided Depth Calibration for Perceptually Optimized Compressive Light Field 3D Display.    (https://ieeexplore.ieee.org/document/8578315)</a>
   Shizheng Wang, Wenjuan Liao, Phil Surman, Zhigang Tu, Yuanjin Zheng, Junsong Yuan   

      * [Zhang-etal2018] A Bi-Directional Message Passing Model for Salient Object Detection.    (https://ieeexplore.ieee.org/document/8578285)</a>
   Lu Zhang, Ju Dai, Huchuan Lu, You He, Gang Wang   

      * [Wang-etal2018] Salient Object Detection Driven by Fixation Prediction.    (https://ieeexplore.ieee.org/document/8578282)</a>
   Wenguan Wang, Jianbing Shen, Xingping Dong, Ali Borji   

      * [Zeng-etal2018] Learning to Promote Saliency Detectors.    (https://ieeexplore.ieee.org/document/8578275)</a>
   Yu Zeng, Huchuan Lu, Lihe Zhang, Mengyang Feng, Ali Borji   

      * [Cheng-etal2018] Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos.    (https://ieeexplore.ieee.org/document/8578252)</a>
   Hsien-Tzu Cheng, Chun-Hung Chao, Jin-Dong Dong, Hao-Kai Wen, Tyng-Luh Liu, Min Sun   

      * [Li-etal2018] Diversity Regularized Spatiotemporal Attention for Video-based Person Re-identification.    (https://ieeexplore.ieee.org/document/8578144)</a>
   Shuang Li, Slawomir Bak, Peter Carr, Xiaogang Wang   

      * [Zhang-etal2018] Progressive Attention Guided Recurrent Network for Salient Object Detection.    (https://ieeexplore.ieee.org/document/8578179)</a>
   Xiaoning Zhang, Tiantian Wang, Jinqing Qi, Huchuan Lu, Gang Wang   

      * [Zhu-etal2018] End-to-end Flow Correlation Tracking with Spatial-temporal Attention.    (https://ieeexplore.ieee.org/document/8578162)</a>
   Zheng Zhu, Wei Wu, Wei Zou, Junjie Yan   

      * [Dolhansky-Ferrer2018] Eye In-Painting with Exemplar Generative Adversarial Networks.    (https://ieeexplore.ieee.org/document/8578922)</a>
   Brian Dolhansky, Cristian Canton Ferrer   

      * [Xu-etal2018] Gaze Prediction in Dynamic 360° Immersive Videos.    (https://ieeexplore.ieee.org/document/8578657)</a>
   Yanyu Xu, Yanbing Dong, Junru Wu, Zhengzhong Sun, Zhiru Shi, Jingyi Yu, Shenghua Gao   

      * [Wang-etal2018] A Hierarchical Generative Model for Eye Image Synthesis and Eye Gaze Estimation.    (https://ieeexplore.ieee.org/document/8578151)</a>
   Kang Wang, Rui Zhao, Qiang Ji   

      * [Vasudevan-etal2018] Object Referring in Videos with Language and Human Gaze.    (https://ieeexplore.ieee.org/document/8578532)</a>
   Arun Balajee Vasudevan, Dengxin Dai, Luc Van Gool   

      * [Fan-etal2018] Inferring Shared Attention in Social Scene Videos.    (https://ieeexplore.ieee.org/document/8578774)
   Fan, Lifeng and Chen, Yixin and Wei, Ping and Wang, Wenguan and Zhu, Song-Chun   

      * [Wei-etal2018] Where and Why Are They Looking? Jointly Inferring Human Attention and Intentions in Complex Tasks.    (https://ieeexplore.ieee.org/document/8578809)
   Ping Wei, Yang Liu, Tianmin Shu, Nanning Zheng, Song-Chun Zhu   

      * [Ranjan-etal2018 CVPRW] Light-weight Head Pose Invariant Gaze Tracking.    (https://ieeexplore.ieee.org/document/8575461)
   Rajeev Ranjan, Shalini De Mello, Jan Kautz   
      * [Palmero-etal2018 BMVC] Recurrent CNN for 3D Gaze Estimation using Appearance and Shape Cues.    (https://www.semanticscholar.org/paper/Recurrent-CNN-for-3D-Gaze-Estimation-using-and-Cues-Palmero-Selva/5fc81eeb3920771984f5824bd4d4524016869f02)
   Cristina Palmero, Javier Selva, Mohammad Ali Bagheri, Sergio Escalera   

      * [Liu-etal2018 BMVC] A Differential Approach for Gaze Estimation with Calibration.    (https://www.semanticscholar.org/paper/A-Differential-Approach-for-Gaze-Estimation-with-Liu-Yu/0310d31020ae59bf3d6ac61b6206dfc0e79b4efe)
   Liu, Gang and Yu, Yu and Funes-Mora, Kenneth A and Odobez, Jean-Marc and SA, Eyeware Tech   

      * [Zhang-etal2017]  Supervision by Fusion: Towards Unsupervised Learning of Deep Salient Object Detector.    (https://ieeexplore.ieee.org/document/8237698)</a>
   Dingwen Zhang, Junwei Han, Yu Zhang   

      * [Wang-etal2017]  A Stagewise Refinement Model for Detecting Salient Objects in Images.    (https://ieeexplore.ieee.org/document/8237695)</a>
   Tiantian Wang, Ali Borji, Lihe Zhang, Pingping Zhang, Huchuan Lu   

      * [He-etal2017]  Delving Into Salient Object Subitizing and Detection.    (https://ieeexplore.ieee.org/document/8237382)</a>
   Shengfeng He, Jianbo Jiao, Xiaodan Zhang, Guoqiang Han, Rynson W.H. Lau   

      * [Chen-etal2017]  Look, Perceive and Segment: Finding the Salient Objects in Images via Two-Stream Fixation-Semantic CNNs.    (https://ieeexplore.ieee.org/document/8237381)</a>
   Xiaowu Chen, Anlin Zheng, Jia Li, Feng Lu   

      * [Zhang-etal2017]  Learning Uncertain Convolutional Features for Accurate Saliency Detection.    (https://ieeexplore.ieee.org/document/8237294)</a>
   Pingping Zhang, Dong Wang, Huchuan Lu, Hongyu Wang, Baocai Yin   

      * [Zhang-etal2017]  Amulet: Aggregating Multi-Level Convolutional Features for Salient Object Detection.    (https://ieeexplore.ieee.org/document/8237293)</a>
   Pingping Zhang, Dong Wang, Huchuan Lu, Hongyu Wang, Xiang Ruan   

      * [Zhu-etal2017]  Saliency Pattern Detection by Ranking Structured Trees.    (https://ieeexplore.ieee.org/document/8237845)</a>
   Lei Zhu, Haibin Ling, Jin Wu, Huiping Deng, Jin Liu   

      * [Leifman-etal2017] Learning Gaze Transitions From Depth to Improve Video Saliency Estimation.    (https://ieeexplore.ieee.org/document/8237450)</a>
   George Leifman, Dmitry Rudoy, Tristan Swedish, Eduardo Bayro-Corrochano, Ramesh Raskar   

      * [Wang-Ji2017] Real Time Eye Gaze Tracking With 3D Deformable Eye-Face Model.    (https://ieeexplore.ieee.org/document/8237376)</a>
   Kang Wang, Qiang Ji   

      * [Deng-Zhu2017] Monocular Free-Head 3D Gaze Tracking with Deep Learning and Geometry Constraints.    (https://ieeexplore.ieee.org/document/8237603)
   Haoping Deng and Wangjiang Zhu   

      * [Recasens-etal2017] Following Gaze in Video.    (https://ieeexplore.ieee.org/document/8237422)
   Adria Recasens and Carl Vondrick and Aditya Khosla and Antonio Torralba   

      * [Shrivastava-etal2017] Learning From Simulated and Unsupervised Images Through Adversarial Training.    (https://ieeexplore.ieee.org/document/8099724)</a>
   Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Joshua Susskind, Wenda Wang, Russell Webb   

      * [Wang-etal2017] Learning to Detect Salient Objects With Image-Level Supervision.    (https://ieeexplore.ieee.org/document/8099887)</a>
   Lijun Wang, Huchuan Lu, Yifan Wang, Mengyang Feng, Dong Wang, Baocai Yin, Xiang Ruan   

      * [Ramanishka-etal2017] Top-Down Visual Saliency Guided by Captions.    (https://ieeexplore.ieee.org/document/8099817)</a>
   Vasili Ramanishka, Abir Das, Jianming Zhang, Kate Saenko   

      * [Luo-etal2017] Non-Local Deep Features for Salient Object Detection.    (https://ieeexplore.ieee.org/document/8100181)</a>
   Zhiming Luo, Akshaya Mishra, Andrew Achkar, Justin Eichel, Shaozi Li, Pierre-Marc Jodoin   

      * [Xia-etal2017] What Is and What Is Not a Salient Object? Learning Salient Object Detector by Ensembling Linear Exemplar Regressors.    (https://ieeexplore.ieee.org/document/8099951)</a>
   Changqun Xia, Jia Li, Xiaowu Chen, Anlin Zheng, Yu Zhang   

      * [Hou-etal2017] Deeply Supervised Salient Object Detection With Short Connections.    (https://ieeexplore.ieee.org/document/8315520)</a>
   Qibin Hou, Ming-Ming Cheng, Xiaowei Hu, Ali Borji, Zhuowen Tu, Philip H. S. Torr   

      * [Li-etal2017] Instance-Level Salient Object Segmentation.    (https://ieeexplore.ieee.org/document/8099517)</a>
   Guanbin Li, Yuan Xie, Liang Lin, Yizhou Yu   

      * [Hu-etal2017] Deep Level Sets for Salient Object Detection.    (https://ieeexplore.ieee.org/document/8099548)</a>
   Ping Hu, Bing Shuai, Jun Liu, Gang Wang   

      * [Tavakoli-etal2017] Saliency Revisited: Analysis of Mouse Movements Versus Fixations.    (https://ieeexplore.ieee.org/document/8100156)</a>
   Hamed R. Tavakoli, Fawad Ahmed, Ali Borji, Jorma Laaksonen   

      * [Karessli-etal2017] Gaze Embeddings for Zero-Shot Image Classification.    (https://ieeexplore.ieee.org/document/8100162)</a>
   Nour Karessli, Zeynep Akata, Bernt Schiele, Andreas Bulling   

      * [Zhang-etal2017] Deep Future Gaze: Gaze Anticipation on Egocentric Videos Using Adversarial Networks.    (https://ieeexplore.ieee.org/document/8099860)
   Mengmi Zhang, Keng Teck Ma, Joo Hwee Lim, Qi Zhao, Jiashi Feng   

      * [Zhang-etal2017 CVPRW] It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation.    (https://ieeexplore.ieee.org/document/8015018)
   Xucong Zhang and Yusuke Sugano and Mario Fritz and Andreas Bulling   

      * [Zhang-etal2017 UIST] Everyday Eye Contact Detection Using Unsupervised Gaze Target Discovery.    (https://dl.acm.org/citation.cfm?id=3126594.3126614)
   Xucong Zhang and Yusuke Sugano and Andreas Bulling   
      * [Wood-etal2016] A 3D Morphable Eye Region Model for Gaze Estimation.    (https://www.semanticscholar.org/paper/A-3D-Morphable-Eye-Region-Model-for-Gaze-Estimation-Wood-Baltrusaitis/c34532fe6bfbd1e6df477c9ffdbb043b77e7804d)
   Erroll Wood and Tadas Baltrusaitis and Louis-Philippe Morency and Peter Robinson and Andreas Bulling   

      * [Yaroslav-etal2016] DeepWarp: Photorealistic Image Resynthesis for Gaze Manipulation.    (https://link.springer.com/chapter/10.1007/978-3-319-46475-6_20)
   Ganin, Yaroslav, Daniil Kononenko, Diana Sungatullina, and Victor Lempitsky   

      * [Jongpil-Pavlovic2016] A Shape-based Approach for Salient Object Detection Using Deep Learning.    (https://link.springer.com/chapter/10.1007/978-3-319-46493-0_28)
   Kim, Jongpil, and Vladimir Pavlovic   

      * [Linzhao-etal2016] Saliency Detection with Recurrent Fully Convolutional Networks.    (https://link.springer.com/chapter/10.1007/978-3-319-46493-0_50)
   Wang, Linzhao, Lijun Wang, Huchuan Lu, Pingping Zhang   

      * [Tiantian-etal2016] Kernelized Subspace Ranking for Saliency Detection.    (https://link.springer.com/chapter/10.1007/978-3-319-46484-8_27)
   Wang, Tiantian, Lihe Zhang, Huchuan Lu, Chong Sun, and Jinqing Qi   

      * [Youbao-Wu2016] Saliency Detection via Combining Region-Level and Pixel-Level Predictions with CNNs.    (https://link.springer.com/chapter/10.1007/978-3-319-46484-8_49)
   Tang, Youbao, and Xiangqian Wu   

      * [Yuqiu-etal2016] Pattern Mining Saliency.    (https://link.springer.com/chapter/10.1007/978-3-319-46466-4_35)
   Kong, Yuqiu, Lijun Wang, Xiuping Liu, Huchuan Lu, and Xiang Ruan   

      * [Zoya-etal2016] Where should saliency models look next?.    (https://link.springer.com/chapter/10.1007/978-3-319-46454-1_49)
   Bylinskii, Zoya, Adrià Recasens, Ali Borji, Aude Oliva, Antonio Torralba, and Frédo Durand   

      * [Aravindh-Vedaldi2016] Salient Deconvolutional Networks.    (https://link.springer.com/chapter/10.1007/978-3-319-46466-4_8)
   Mahendran, Aravindh, and Andrea Vedaldi   
      * [Krafka-etal2016] Eye Tracking for Everyone.    (https://ieeexplore.ieee.org/document/7780608)
   Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba   

      * [Yu-etal2016] Learning Reconstruction-Based Remote Gaze Estimation.    (https://ieeexplore.ieee.org/document/7780744)
   Pei Yu and Jiahuan Zhou and Ying Wu   

      * [Kruthiventi-etal2016] Saliency Unified: A Deep Architecture for Simultaneous Eye Fixation Prediction and Salient Object Segmentation.    (https://ieeexplore.ieee.org/document/7780992)
   Srinivas S. S. Kruthiventi, Vennela Gudisa, Jaley H. Dholakiya, R. Venkatesh Babu   

      * [Volokitin-etal2016] Predicting When Saliency Maps Are Accurate and Eye Fixations Consistent.    (https://ieeexplore.ieee.org/document/7780434)
   Anna Volokitin, Michael Gygli, Xavier Boix   

      * [Kuen-etal2016] Recurrent Attentional Networks for Saliency Detection.    (https://ieeexplore.ieee.org/document/7780768)
   Jason Kuen, Zhenhua Wang, Gang Wang   

      * [Cholakkal-etal2016] Backtracking ScSPM Image Classifier for Weakly Supervised Top-Down Saliency.    (https://ieeexplore.ieee.org/document/7780939)
   Hisham Cholakkal, Jubin Johnson, Deepu Rajan   

      * [He-Lau2016] Exemplar-Driven Top-Down Saliency Detection via Deep Association.    (https://ieeexplore.ieee.org/document/7780986)
   Shengfeng He, Rynson W.H. Lau   

      * [Zhang-etal2016] Unconstrained Salient Object Detection via Proposal Subset Optimization.    (https://ieeexplore.ieee.org/document/7780987)
   Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price, Radomir Mech   

      * [Jetley-etal2016] End-To-End Saliency Mapping via Probability Distribution Prediction.    (https://ieeexplore.ieee.org/document/7780989)
   Saumya Jetley, Naila Murray, Eleonora Vig   

      * [Li-Yu2016] Deep Contrast Learning for Salient Object Detection.    (https://ieeexplore.ieee.org/document/7780427)
   Guanbin Li, Yizhou Yu   

      * [Bruce-etal2016] A Deeper Look at Saliency: Feature Contrast, Semantics, and Beyond.    (https://ieeexplore.ieee.org/document/7780431)
   Neil D. B. Bruce, Christopher Catton, Sasa Janjic   

      * [Wloka-Tsotsos2016] Spatially Binned ROC: A Comprehensive Saliency Metric.    (https://ieeexplore.ieee.org/document/7780432)
   Calden Wloka, John Tsotsos   

      * [Pan-etal2016] Shallow and Deep Convolutional Networks for Saliency Prediction.    (https://ieeexplore.ieee.org/document/7780440)
   Junting Pan, Elisa Sayrol, Xavier Giro-i-Nieto, Kevin McGuinness, Noel E. O'Connor   

      * [Lee-etal2016] Deep Saliency With Encoded Low Level Distance Map and High Level Features.    (https://ieeexplore.ieee.org/document/7780447)
   Gayoung Lee, Yu-Wing Tai, Junmo Kim   

      * [Liu-Han2016] DHSNet: Deep Hierarchical Saliency Network for Salient Object Detection.    (https://ieeexplore.ieee.org/document/7780449)
   Nian Liu, Junwei Han   

      * [Tu-etal2016] Real-Time Salient Object Detection With a Minimum Spanning Tree.    (https://ieeexplore.ieee.org/document/7780625)
   Wei-Chih Tu, Shengfeng He, Qingxiong Yang, Shao-Yi Chien   

      * [Feng-etal2016] Local Background Enclosure for RGB-D Salient Object Detection.    (https://ieeexplore.ieee.org/document/7780626)
   David Feng, Nick Barnes, Shaodi You, Chris McCarthy   

      * [Jeni-Cohn2016 CVPRW] Person-Independent 3D Gaze Estimation Using Face Frontalization.    (https://ieeexplore.ieee.org/document/7789594)
   Laszlo A. Jeni and Jeffrey F. Cohn   
      * [Wood-etal2016 ETRA] Learning an appearance-based gaze estimator from one million synthesised images.    (https://dl.acm.org/citation.cfm?id=2857492)
   Erroll Wood and Tadas Baltrušaitis and Louis-Philippe Morency and Peter Robinson and Andreas Bulling   

      * [Tonsen-etal2016 ETRA] Labelled pupils in the wild: a dataset for studying pupil detection in unconstrained environments.    (https://dl.acm.org/citation.cfm?id=2857520)
   Marc Tonsen and Xucong Zhang and Yusuke Sugano and Andreas Bulling   

      * [Ghiass-Arandjelovic2016 IJCAI] Highly accurate gaze estimation using a consumer RGB-depth sensor.    (https://dl.acm.org/citation.cfm?id=3061092)
   Reza Shoja Ghiass and Ognjen Arandjelovic   
      * [Wood-etal2015] Rendering of Eyes for Eye-Shape Registration and Gaze Estimation.    (https://ieeexplore.ieee.org/document/7410785)
   Erroll Wood and Tadas Baltruaitis and Xucong Zhang and Yusuke Sugano and Peter Robinson and Andreas Bullingo   

      * [Zhang-etal2015] Appearance-based gaze estimation in the wild.    (https://ieeexplore.ieee.org/document/7299081)
   Xucong Zhang and Yusuke Sugano and Mario Fritz and Andreas Bulling   


      * [Wang-etal2015 ISKE] A Survey on Gaze Estimation.    (https://ieeexplore.ieee.org/document/7383057)
   Xiaomeng Wang and Kang Liu and Xu Qian   

      * [Sugano-etal2014] Learning-by-Synthesis for Appearance-Based 3D Gaze Estimation.    (https://ieeexplore.ieee.org/document/6909631)
   Yusuke Sugano and Yasuyuki Matsushita and Yoichi Sato   

      * [Mora-Odobez2014] Geometric Generative Gaze Estimation (G3E) for Remote RGB-D Cameras.    (https://ieeexplore.ieee.org/document/6909625)
   Kenneth Alberto Funes Mora and Jean-Marc Odobez   
     * [Mora-etal2014 ETRA] EYEDIAP: a database for the development and evaluation of gaze estimation algorithms from RGB and RGB-D cameras.    (https://dl.acm.org/citation.cfm?id=2578190)
   Kenneth Alberto Funes Mora and Florent Monay and Jean-Marc Odobez   

      * [Jianfeng-Shigang2014 CVPRW] Eye-Model-Based Gaze Estimation by RGB-D Camera.    (https://ieeexplore.ieee.org/document/6910042)
   Xucong Zhang and Yusuke Sugano and Mario Fritz and Andreas Bulling   

      * [Cazzato-etal2014 ICPRW] Pervasive Retail Strategy Using a Low-Cost Free Gaze Estimation System.    (https://link.springer.com/chapter/10.1007/978-3-319-12811-5_2)
   Dario Cazzato and Marco Leo and Paolo Spagnolo and Cosimo Distante   

      * [Schneider-etal2014 ICPR] Manifold Alignment for Person Independent Appearance-Based Gaze Estimation.    (https://ieeexplore.ieee.org/document/6976920/)
   Timo Schneider and Boris Schauerte and Rainer Stiefelhagen   

      * [Kassner-etal2014 UbiComp] Pupil: an open source platform for pervasive eye tracking and mobile gaze-based interaction.    (https://dl.acm.org/citation.cfm?id=2641695)
   Moritz Kassner and William Patera and Andreas Bulling   
      * [Alnajar-etal2013] Calibration-Free Gaze Estimation Using Human Gaze Patterns.    (https://ieeexplore.ieee.org/document/6751126)
   Fares Alnajar and Theo Gevers and Roberto Valenti and Sennay Ghebreab   

      * [Lu-etal2011a] Inferring human gaze from appearance via adaptive linear regression.    (https://ieeexplore.ieee.org/document/6126237)
   Feng Lu and Yusuke Sugano and Takahiro Okabe and Yoichi Sato   

      * [Wang-etal2003] Eye gaze estimation from a single image of one eye.    (https://ieeexplore.ieee.org/document/1238328)
   Wang and Sung and Ronda Venkateswarlu   
      * [Sugano-etal2008] An Incremental Learning Method for Unconstrained Gaze Estimation.    (https://link.springer.com/chapter/10.1007%2F978-3-540-88690-7_49)
   Yusuke Sugano and Yasuyuki Matsushita and Yoichi Sato and Hideki Koike   

      * [Baltrusaitis-etal2012] 3D Constrained Local Model for rigid and non-rigid facial tracking.    (https://ieeexplore.ieee.org/document/6247980)
   T. Baltrusaitis and P. Robinson and L. Morency   

      * [Mora-Odobez2012 CVPRW] Gaze estimation from multimodal Kinect data.    (https://ieeexplore.ieee.org/document/6239182)
   Kenneth Alberto Funes Mora and Jean-Marc Odobez   

      * [Chen-Ji2011] Probabilistic gaze estimation without active personal calibration.    (https://ieeexplore.ieee.org/document/5995675)
   Jixu Chen and Qiang Ji   

      * [Sugano-etal2010] Calibration-free gaze sensing using saliency maps.    (https://ieeexplore.ieee.org/document/5539984)
   Yusuke Sugano and Yasuyuki Matsushita and Yoichi Sato   

      * [Williams-etal2006] Sparse and Semi-supervised Visual Mapping with the S^3GP.    (https://ieeexplore.ieee.org/document/1640764)
   O. Williams and A. Blake and R. Cipolla   

      * [Zhu-Ji2005] Eye Gaze Tracking under Natural Head Movements.    (https://ieeexplore.ieee.org/document/1467364)
   Zhiwei Zhu and Qiang Ji   

      * [Choi-etal2013 URAI] Appearance-based gaze estimation using kinect.    (https://ieeexplore.ieee.org/document/6677362)
   Jinsoo Choi and Byungtae Ahn and Jaesik Parl and In So Kweon   

      * [Mora-Odobez2013 ICIP] Person independent 3D gaze estimation from remote RGB-D cameras.    (https://ieeexplore.ieee.org/document/6738574)
   Kenneth Alberto Funes Mora and Jean-Marc Odobez   

      * [Liang-etal2013 ETSA] Appearance-based gaze tracking with spectral clustering and semi-supervised Gaussian process regression.    (https://ieeexplore.ieee.org/document/6677362)
   Ke Liang and Youssef Chahir and Michèle Molina and Charles Tijus and François Jouen   

      * [McMurrough-etal2012 ETRA] An eye tracking dataset for point of gaze detection.    (https://dl.acm.org/citation.cfm?id=2168622)
   Christopher D. McMurrough and Vangelis Metsis and Jonathan Rich and Fillia Makedon   

      * [Lu-etal2012 ICPR] Head pose-free appearance-based gaze sensing via eye image synthesis.    (https://ieeexplore.ieee.org/document/6460306)
   Feng Lu and Yusuke Sugano and Takahiro Okabe and Yoichi Sato   

      * [Mohammadi-Raie2012 ICEE] Robust pose-invariant eye gaze estimation using geometrical features of iris and pupil images.    (https://ieeexplore.ieee.org/document/6292425)
   Mohammad Reza Mohammadi and Abolghasem Raie   

      * A Head Pose-free Approach for Appearance-based Gaze Estimation. (http://www.bmva.org/bmvc/2011/proceedings/paper126/index.html)
   Feng Lu and Takahiro Okabe and Yusuke Sugano and Yoichi Sato   

      * Social interaction discovery by statistical analysis of F-formations.    (http://www.bmva.org/bmvc/2011/proceedings/paper23/index.html)
   Marco Cristani and Loris Bazzani and Giulia Paggetti and Andrea Fossati and Diego Tosato and Alessio Del Bue and Gloria Menegaz and Vittorio Murino   

* [Effective eye-gaze input into Windows.](https://dl.acm.org/citation.cfm?id=355021)
   Chris Lankford   
  
    --->



