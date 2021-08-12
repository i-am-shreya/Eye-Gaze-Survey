# EyeGazeSurvey
Automatic Gaze Analysis: A Survey of Deep Learning based Approaches by Shreya Ghosh, Abhinav Dhall, Munawar Hayat, Jarrod Knibbe and Qiang Ji. [Link](https://drive.google.com/file/d/1T4v8tAeicntSkzTZFX8-26IDXH0UYl1Z/view?usp=sharing)

  
If we miss your work, please let us know and we'll add it. 

## Contact
- <a href="https://sites.google.com/view/shreyaghosh/home">Shreya Ghosh</a>.

<!---## Update-->

## Datasets
A comparison of gaze datasets with respect to several attributes (i.e. number of subjects (\# subjects), gaze labels, modality, headpose and gaze angle in yaw and pitch axis, environment (Env.), baseline method, data statistics (\# of data), and year of publication.) The abbreviations used are: In: Indoor, Out: Outdoor, Both: Indoor + Outdoor, Gen.: Generation, u/k: unknown, Seq.: Sequence, VF: Visual Field, EB: Eye Blink, GE: Gaze Event, GBRT: Gradient Boosting Regression Trees, GC: Gaze Communication, GNN: Graph Neural Network and Seg.: Segmentation.
![datasets](/images/datasets.png)


## Gaze Analysis Methods
A comparison of gaze analysis methods with respect to registration (Reg.), representation (Represent.), Level of Supervision, Model, Prediction, validation on benchmark datasets (validation), Platforms, Publication venue (Publ.) and year. Here, GV: Gaze Vector, Scr.: Screen, LOSO: Leave One Subject Out, LPIPS: Learned Perceptual Image Patch Similarity, MM: Morphable Model, RRF: Random Regression Forest, AEM: Anatomic Eye Model, GRN: Gaze Regression Network, ET: External Target, FV: Free Viewing, HH: HandHeld Device, HMD: Head Mounted Device, Seg.: Segmentation and GR: Gaze Redirection, LAEO: Looking At Each Other.
![datasets](/images/prior_work.png)

# Awesome Eye-Gaze: [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
A curated list of papers and datsets for various gaze estimation techniques, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

## Contents
 - [Eye Gaze Estimation](#Eye-Gaze-Estimation)
 - [Gaze Trajectory](#Gaze-Trajectory)
 - [Gaze Redirection](#Gaze-Redirection)
 - [Gaze Zone](#Gaze-Zone)
 - [Looking At Each Other](#LAEO) 
 - [Uncategorized Papers](#Uncategorized-Papers)

<!---- [Cross-modal Headpose (Audio-Video / Video-Audio)](#Cross-modal-Generation-(Audio-Video--Video-Audio)) [Multi-modal Architectures](#Multi-modal-Architectures)-->

#### Eye Gaze Estimation
* [How is Gaze Influenced by Image Transformations? Dataset and Model.](https://doi.org/10.1109/TIP.2019.2945857) - Zhaohui Che, Ali Borji, Guangtao Zhai, Xiongkuo Min, Guodong Guo, Patrick Le Callet (2020 IEEE TIP) [[code]](https://github.com/CZHQuality/Sal-CFS-GAN)
* [Low Cost Gaze Estimation: Knowledge-Based Solutions.](https://doi.org/10.1109/TIP.2019.2946452) - Ion Martinikorena, Andoni Larumbe-Bergera, Mikel Ariz, Sonia Porta, Rafael Cabeza, Arantxa Villanueva (2020 IEEE TIP)
* [Gaze Estimation by Exploring Two-Eye Asymmetry.] (https://doi.org/10.1109/TIP.2020.2982828)- Yihua Cheng, Xucong Zhang, Feng Lu, Yoichi Sato (2020 IEEE TIP)
* [MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation.] (https://ieeexplore.ieee.org/document/8122058) - Xucong Zhang and Yusuke Sugano and Mario Fritz and Andreas Bulling (2017 IEEE TPAMI) 
* [Appearance-Based Gaze Estimation via Uncalibrated Gaze Pattern Recovery.](https://ieeexplore.ieee.org/document/7833091) - Feng Lu and Xiaowu Chen and Yoichi Sato  

   [Tonsen-etal2017 IMWUT] InvisibleEye: Mobile Eye Tracking Using Multiple Low-Resolution Cameras and Learning-Based Gaze Estimation.   (https://dl.acm.org/citation.cfm?id=3130971  )  
  Marc Tonsen and Julian Steil and Yusuke Sugano and Andreas Bulling  

   [Cazzato-etal2016 PRL] A low-cost and calibration-free gaze estimator for soft biometrics: An explorative study.   (https://www.sciencedirect.com/science/article/pii/S0167865515003669  )  
  Dario Cazzato and Andrea Evangelista and Marco Leo and Pierluigi Carcagní and Cosimo Distante  

   [Cristina-Camilleri2016 CVIU] Model-based head pose-free gaze estimation for assistive communication.   (https://www.sciencedirect.com/science/article/pii/S1077314216000667  )  
  Stefania Cristina and Kenneth P. Camilleri  

   [George-Routray2016 IET CV] Fast and accurate algorithm for eye localisation for gaze tracking in low-resolution images.   (http://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2015.0316  )  
  Anjith George and Aurobinda Routray  

   [Mora-Odobez2015 VisRes] Gaze Estimation in the 3D Space Using RGB-D Sensors.   (http://publications.idiap.ch/index.php/publications/show/3228  )  
  Kenneth A. Funes-Mora and Jean-Marc Odobez  

   [Parks-etal2015 VisRes] Augmented saliency model using automatic 3D head pose detection and learned gaze following in natural scenes.   (https://www.sciencedirect.com/science/article/pii/S0042698914002739  )  
  Daniel Parks and Ali Borji and Laurent Itti  

   [Sugano-etal2015 THMS] Appearance-Based Gaze Estimation With Online Calibration From Mouse Operations.   (https://ieeexplore.ieee.org/document/7050250  )  
  Yusuke Sugano and Yasuyuki Matsushita and Yoichi Sato and Hideki Koike  

   [Fridman-etal2015 MIS] Driver Gaze Region Estimation Without Using Eye Movement.   (https://ieeexplore.ieee.org/document/7478592  )  
  Lex Fridman and Philipp Langhans and Joonbum Lee and Bryan Reimer  

   [Gwon-etal2014 Sensors] Gaze Tracking System for User Wearing Glasses.   (https://www.mdpi.com/1424-8220/14/2/2110  )  
  Su Gwon and Chul Cho and Hyeon Lee and Won Lee and Kang Park  

   [Lu-etal2014 TPAMI] Adaptive Linear Regression for Appearance-Based Gaze Estimation.   (https://ieeexplore.ieee.org/document/6777326  )  
  Feng Lu and Yusuke Sugano and Takahiro Okabe and Yoichi Sato  

   [Lu-etal2014a IVC] Learning gaze biases with head motion for head pose-free gaze estimation.   (https://www.sciencedirect.com/science/article/pii/S0262885614000171  )  
  Feng Lu and Takahiro Okabe and Yusuke Sugano and Yoichi Sato  

   [Sun-etal2014 TMM] Real-Time Gaze Estimation with Online Calibration.   (https://ieeexplore.ieee.org/document/6916495  )  
  Li Sun and Mingli Song and Zicheng Liu and Ming-Ting Sun  

   [Sugano-etal2013 TPAMI] Appearance-Based Gaze Estimation Using Visual Saliency.   (https://ieeexplore.ieee.org/document/6193107  )  
  Yusuke Sugano and Yasuyuki Matsushita and Yoichi Sato  

   [Gwon-etal2013 IJARS] Robust Eye and Pupil Detection Method for Gaze Tracking.   (journals.sagepub.com/doi/full/10.5772/55520  )  
  Su Yeong Gwon and Chul Woo Cho and Hyeon Chang Lee and Won Oh Lee and Kang Ryoung Park  

   [Corcoran-etal2012 TCE] Real-time eye gaze tracking for gaming design and consumer electronics systems.   (https://ieeexplore.ieee.org/document/6227433  )  
  Peter Corcoran and Florin Nanu and Stefan Petrescu and Petronel Bigioi  

   [Valenti-etal2012 TIP] Combining Head Pose and Eye Location Information for Gaze Estimation.   (https://ieeexplore.ieee.org/document/5959981  )  
  R. Valenti and N. Sebe and T. Gevers  

   [Lee-etal2010 TCE] Gaze tracking system at a distance for controlling IPTV. <a https://ieeexplore.ieee.org/document/5681143  )  
  Hyeon Lee and Duc Luong and Chul Cho and Eui Lee and Kang Park  

   [Morimoto-Mimica2005 CVIU] Eye gaze tracking techniques for interactive applications.   (https://www.sciencedirect.com/science/article/pii/S1077314204001109  )  
  Carlos H. Morimoto and Marcio R.M. Mimica  


   [Fang-etal2021] Dual Attention Guided Gaze Target Detection in the Wild  (https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_Dual_Attention_Guided_Gaze_Target_Detection_in_the_Wild_CVPR_2021_paper.pdf">[PDF]  
  Yi Fang, Jiapeng Tang, Wang Shen, Wei Shen, Xiao Gu, Li Song, Guangtao Zhai  
   [Zhong-etal2021] Glance and Gaze: Inferring Action-Aware Points for One-Stage Human-Object Interaction Detection  (https://openaccess.thecvf.com/content/CVPR2021/papers/Zhong_Glance_and_Gaze_Inferring_Action-Aware_Points_for_One-Stage_Human-Object_Interaction_CVPR_2021_paper.pdf">[PDF]  
  Xubin Zhong, Xian Qu, Changxing Ding, Dacheng Tao  
   [Kothari-etal2021] Weakly-Supervised Physically Unconstrained Gaze Estimation  (https://openaccess.thecvf.com/content/CVPR2021/papers/Kothari_Weakly-Supervised_Physically_Unconstrained_Gaze_Estimation_CVPR_2021_paper.pdf">[PDF]  
  Rakshit Kothari, Shalini De Mello, Umar Iqbal, Wonmin Byeon, Seonwook Park, Jan Kautz  
   [Liu-etal2021] Goal-Oriented Gaze Estimation for Zero-Shot Learning  (https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Goal-Oriented_Gaze_Estimation_for_Zero-Shot_Learning_CVPR_2021_paper.pdf">[PDF]  
  Yang Liu, Lei Zhou, Xiao Bai, Yifei Huang, Lin Gu, Jun Zhou, Tatsuya Harada  

   [Meng-etal2021] Connecting What To Say With Where To Look by Modeling Human Attention Traces  (https://openaccess.thecvf.com/content/CVPR2021/papers/Meng_Connecting_What_To_Say_With_Where_To_Look_by_Modeling_CVPR_2021_paper.pdf">[PDF]  
  Zihang Meng, Licheng Yu, Ning Zhang, Tamara L. Berg, Babak Damavandi, Vikas Singh, Amy Bearman  

   [Tomas-etal2021 CVPRW] GOO: A Dataset for Gaze Object Prediction in Retail Environments  (https://arxiv.org/pdf/2105.10793">[PDF]  
  	Henri Tomas, Marcus Reyes, Raimarc Dionido, Mark Ty, Jonric Mirando, Joel Casimiro, Rowel Atienza, Richard Guinto  
   [Poulopoulos-etal2021 CVPRW] PupilTAN: A Few-Shot Adversarial Pupil Localizer  (https://openaccess.thecvf.com/content/CVPR2021W/GAZE/papers/Poulopoulos_PupilTAN_A_Few-Shot_Adversarial_Pupil_Localizer_CVPRW_2021_paper.pdf">[PDF]  
  	Nikolaos Poulopoulos, Emmanouil Z. Psarakis, Dimitrios Kosmopoulos  
   [Murthy-etal2021 CVPRW] Appearance-Based Gaze Estimation Using Attention and Difference Mechanism  (https://openaccess.thecvf.com/content/CVPR2021W/GAZE/papers/D_Appearance-Based_Gaze_Estimation_Using_Attention_and_Difference_Mechanism_CVPRW_2021_paper.pdf">[PDF]  
  	Murthy L R D, Pradipta Biswas  
   [Siegfried-etal2021 CVPRW] Visual Focus of Attention Estimation in 3D Scene With an Arbitrary Number of Targets  (https://openaccess.thecvf.com/content/CVPR2021W/GAZE/papers/Siegfried_Visual_Focus_of_Attention_Estimation_in_3D_Scene_With_an_CVPRW_2021_paper.pdf">[PDF]  
  	Remy Siegfried, Jean-Marc Odobez  



   [Cheng-etal2021 arXiv] Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark  (https://arxiv.org/pdf/2104.12668">[PDF]  
  	Yihua Cheng, Haofei Wang, Yiwei Bao, Feng Lu  
   [Cheng-etal2021 arXiv] PureGaze: Purifying Gaze Feature for Generalizable Gaze Estimation  (https://arxiv.org/pdf/2103.13173">[PDF]  
  	Yihua Cheng, Yiwei Bao, Feng Lu  
   [Cai-etal2021 arXiv] Gaze Estimation with an Ensemble of Four Architectures  (https://arxiv.org/pdf/2107.01980">[PDF]  
  	Xin Cai, Boyu Chen, Jiabei Zeng, Jiajun Zhang, Yunjia Sun, Xiao Wang, Zhilong Ji, Xiao Liu, Xilin Chen, Shiguang Shan  
   [Bao-etal2021 arXiv] The Story in Your Eyes: An Individual-difference-aware Model for Cross-person Gaze Estimation  (https://arxiv.org/pdf/2107.01980">[PDF]  
  	Jun Bao, Buyu Liu, Jun Yu  
   [Ji-etal2021 arXiv] Bayesian Eye Tracking  (https://arxiv.org/pdf/2106.13387">[PDF]  
  	Qiang Ji, Kang Wang  
   [Jindal-etal2021 arXiv] CUDA-GR: Controllable Unsupervised Domain Adaptation for Gaze Redirection  (https://arxiv.org/pdf/2106.10852">[PDF]  
  	Swati Jindal, Xin Eric Wang  
   [Yu-etal2021 arXiv] Glance-and-Gaze Vision Transformer  (https://arxiv.org/pdf/2106.02277">[PDF]  
  	Qihang Yu, Yingda Xia, Yutong Bai, Yongyi Lu, Alan Yuille, Wei Shen  
   [Cheng-etal2021 arXiv] Gaze Estimation using Transformer  (https://arxiv.org/pdf/2105.14424">[PDF]  
  	Yihua Cheng, Feng Lu  




   [Zhang-etal2020] ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation.  (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500358.pdf">[PDF]  
  Xucong Zhang, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang , Otmar Hilliges   

   [Park-etal2020] Towards End-to-end Video-based Eye-Tracking.  (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570732.pdf">[PDF]  
  Seonwook Park, Emre Aksan, Xucong Zhang, Otmar Hilliges   


   [Lee-etal2020] Deep Learning-based Pupil Center Detection for Fast and Accurate Eye Tracking System.  (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640035.pdf">[PDF]  
  Kang Il Lee, Jung Ho Jeon, Byung Cheol Song   

<h3><a id="user-content-2019-cvpr" class="anchor" aria-hidden="true" href="#2020-cvpr"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2020 CVPR</h3>

   [Yu-etal2020] Unsupervised Representation Learning for Gaze Estimation.  (https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Unsupervised_Representation_Learning_for_Gaze_Estimation_CVPR_2020_paper.pdf">[PDF]  
  Yu Yu, Jean-Marc Odobez   

   [Pal-etal2020] "Looking at the Right Stuff" - Guided Semantic-Gaze for Autonomous Driving.  (https://openaccess.thecvf.com/content_CVPR_2020/papers/Pal_Looking_at_the_Right_Stuff_-_Guided_Semantic-Gaze_for_Autonomous_CVPR_2020_paper.pdf">[PDF]  
  Anwesan Pal, Sayan Mondal, Henrik I. Christensen   


   [Fang-etal2020 ICIP] Identifying Children with Autism Spectrum Disorder Based on Gaze-Following.  (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9190831">[PDF]  
  	Yi Fang, Huiyu Duan, Fangyu Shi, Xiongkuo Min, Guangtao Zhai  

   [Guo-etal2020 ACCV] Domain Adaptation Gaze Estimation by Embedding with Prediction Consistency.  (https://openaccess.thecvf.com/content/ACCV2020/papers/Guo_Domain_Adaptation_Gaze_Estimation_by_Embedding_with_Prediction_Consistency_ACCV_2020_paper.pdf">[PDF]  
  Zidong Guo, Zejian Yuan, Chong Zhang, Wanchao Chi, Yonggen Ling, Shenghao Zhang   

   [Chen-etal2020 WACV] Offset Calibration for Appearance-Based Gaze Estimation via Gaze Decomposition.  (https://openaccess.thecvf.com/content_WACV_2020/papers/Chen_Offset_Calibration_for_Appearance-Based_Gaze_Estimation_via_Gaze_Decomposition_WACV_2020_paper.pdf">[PDF]  
  Zhaokang Chen, Bertram Shi   

   [Dias-etal2020 WACV] Gaze Estimation for Assisted Living Environments.  (https://openaccess.thecvf.com/content_WACV_2020/papers/Dias_Gaze_Estimation_for_Assisted_Living_Environments_WACV_2020_paper.pdf">[PDF]  
  Philipe Ambrozio Dias, Damiano Malafronte, Henry Medeiros, Francesca Odone   

   [Kaur-etal2020 WACV] EyeGAN: Gaze-Preserving, Mask-Mediated Eye Image Synthesis.  (https://openaccess.thecvf.com/content_WACV_2020/papers/Kaur_EyeGAN_Gaze-Preserving_Mask-Mediated_Eye_Image_Synthesis_WACV_2020_paper.pdf">[PDF]  
  Harsimran Kaur, Roberto Manduchi  

   [Wang-etal2020 WACV] Learning to Detect Head Movement in Unconstrained Remote Gaze Estimation in the Wild.  (https://openaccess.thecvf.com/content_WACV_2020/papers/Wang_Learning_to_Detect_Head_Movement_in_Unconstrained_Remote_Gaze_Estimation_WACV_2020_paper.pdf">[PDF]  
  Zhecan Wang, Jian Zhao, Cheng Lu, Fan Yang, Han Huang, lianji li, Yandong Guo  

   [Zhang-etal2020 BMVC] Learning-based Region Selection for End-to-End Gaze Estimation.  (https://www.bmvc2020-conference.com/assets/papers/0086.pdf">[PDF]  
  Xucong Zhang, Yusuke Sugano, Andreas Bulling and Otmar Hilliges  

   [Sood-etal2020 NeurIPS] Improving Natural Language Processing Tasks with Human Gaze-Guided Neural Attention.  (https://papers.nips.cc/paper/2020/file/460191c72f67e90150a093b4585e7eb4-Paper.pdf">[PDF]  
  Ekta Sood, Simon Tannert, Philipp Mueller, Andreas Bulling  

   [Zheng-etal2020 NeurIPS] Self-Learning Transformations for Improving Gaze and Head Redirection.  (https://papers.nips.cc/paper/2020/file/98f2d76d4d9caf408180b5abfa83ae87-Paper.pdf">[PDF]  
  Yufeng Zheng, Seonwook Park, Xucong Zhang, Shalini De Mello, Otmar Hilliges  

   [Zheng-etal2020 NeurIPS] Dynamic Fusion of Eye Movement Data and Verbal Narrations in Knowledge-rich Domains.  (https://papers.nips.cc/paper/2020/file/16837163fee34175358a47e0b51485ff-Paper.pdf">[PDF]  
  Ervine Zheng, Qi Yu, Rui Li, Pengcheng Shi, Anne Haake  

   [Orhan-etal2020 NeurIPS] Self-supervised learning through the eyes of a child.  (https://papers.nips.cc/paper/2020/file/7183145a2a3e0ce2b68cd3735186b1d5-Paper.pdf">[PDF]  
  Emin Orhan, Vaibhav Gupta, Brenden M. Lake  
   [Cheng-etal2020 AAAI] A Coarse-to-Fine Adaptive Network for Appearance-Based Gaze Estimation.  (https://aaai.org/ojs/index.php/AAAI/article/view/6636/6490">[PDF]  
  Yihua Cheng, Shiyao Huang, Fei Wang, Chen Qian, Feng Lu  


<h3><a id="user-content-2019-cvpr" class="anchor" aria-hidden="true" href="#2019-iccv"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2019 ICCV</h3>

   [Fan-etal2019] Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning.   (http://openaccess.thecvf.com/content_ICCV_2019/papers/Fan_Understanding_Human_Gaze_Communication_by_Spatio-Temporal_Graph_Reasoning_ICCV_2019_paper.pdf">[PDF]  
  Lifeng Fan, Wenguan Wang, Siyuan Huang, Xinyu Tang, Song-Chun Zhu   

   [Kellnhofer-etal2019] Gaze360: Physically Unconstrained Gaze Estimation in the Wild.   (http://openaccess.thecvf.com/content_ICCV_2019/papers/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.pdf">[PDF]  
  Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Matusik, Antonio Torralba   

   [He-etal2019] Photo-Realistic Monocular Gaze Redirection Using Generative Adversarial Networks.   (http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Photo-Realistic_Monocular_Gaze_Redirection_Using_Generative_Adversarial_Networks_ICCV_2019_paper.pdf">[PDF]  
  Zhe He, Adrian Spurr, Xucong Zhang, Otmar Hilliges   

   [Park-etal2019] Few-Shot Adaptive Gaze Estimation.   (http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf">[PDF]  
  Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges, Jan Kautz   
<h3><a id="user-content-2019-cvpr" class="anchor" aria-hidden="true" href="#2019-cvpr"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2019 CVPR</h3>

   [Xiong-etal2019] Mixed Effects Neural Networks (MeNets) With Applications to Gaze Estimation.   (">[PDF]  
  Yunyang Xiong, Hyunwoo J. Kim, Vikas Singh  

   [Yu-etal2019] Improving Few-Shot User-Specific Gaze Adaptation via Gaze Redirection Synthesis.   (http://www.idiap.ch/~odobez/publications/YuLiuOdobez-CVPR2019.pdf">[PDF]  
  Yu Yu, Gang Liu, Jean-Marc Odobez  

   [Wang-etal2019] Neuro-Inspired Eye Tracking With Eye Movement Dynamics.   (http://homepages.rpi.edu/~wangk10/papers/wang2019neural.pdf">[PDF]  
  Kang Wang, Hui Su, Qiang Ji  

   [Wang-etal2019] Generalizing Eye Tracking With Bayesian Adversarial Learning.   (https://www.semanticscholar.org/paper/Generalizing-Eye-Tracking-with-Bayesian-Adversarial-Wang-Zhao/77b9b6786699a236aad0c3fa3734730ece4a780f">[PDF]  
  Kang Wang, Rui Zhao, Hui Su, Qiang Ji  

<h3><a id="user-content-2019-others" class="anchor" aria-hidden="true" href="#2019-others"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2019 Others</h3>

   [Hickson-etal2019 WACV] Eyemotion: Classifying Facial Expressions in VR Using Eye-Tracking Cameras.   (https://doi.org/10.1109/WACV.2019.00178">[PDF]  
  Steven Hickson, Nick Dufour, Avneesh Sud, Vivek Kwatra, Irfan A. Essa  

   [Lian-etal2019 AAAI] RGBD Based Gaze Estimation via Multi-Task CNN.  (https://doi.org/10.1609/aaai.v33i01.33012488">[PDF]  
  Dongze Lian, Ziheng Zhang, Weixin Luo, Lina Hu, Minye Wu, Zechao Li, Jingyi Yu, Shenghua Gao  


<h3><a id="user-content-2018-eccv" class="anchor" aria-hidden="true" href="#2018-eccv"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2018 ECCV</h3>

   [Fischer-etal2018] RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments.   (https://link.springer.com/chapter/10.1007%2F978-3-030-01249-6_21  )  
  Tobias Fischer, Hyung Jin Chang, Yiannis Demiris  

   [Yohanandan-etal2018] Saliency Preservation in Low-Resolution Grayscale Images.   (https://link.springer.com/chapter/10.1007/978-3-030-01231-1_15">[PDF]  
  Shivanthan Yohanandan, Andy Song, Adrian G. Dyer, Dacheng Tao  

   [Jiang-etal2018] DeepVS: A Deep Learning Based Video Saliency Prediction Approach.   (https://link.springer.com/chapter/10.1007/978-3-030-01264-9_37">[PDF]  
  Lai Jiang, Mai Xu, Tie Liu, Minglang Qiao, Zulin Wang  

   [Kummerer-etal2018] Saliency Benchmarking Made Easy: Separating Models, Maps and Metrics.   (https://link.springer.com/chapter/10.1007/978-3-030-01270-0_47">[PDF]  
  Matthias Kummerer, Thomas S. A. Wallis, Matthias Bethge  

   [Zheng-etal2018] Task-driven Webpage Saliency.   (https://link.springer.com/chapter/10.1007/978-3-030-01264-9_18">[PDF]  
  Quanlong Zheng, Jianbo Jiao, Ying Cao, Rynson W.H. Lau  

   [Zhang-etal2018] Saliency Detection in 360° Videos.   (https://link.springer.com/chapter/10.1007/978-3-030-01234-2_30">[PDF]  
  Ziheng Zhang, Yanyu Xu, Jingyi Yu, Shenghua Gao  

   [Song-etal2018] Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection.   (https://link.springer.com/chapter/10.1007/978-3-030-01252-6_44  )  
  Hongmei Song, Wenguan Wang, Sanyuan Zhao, Jianbing Shen, Kin-Man Lam  

   [Fan-etal2018] Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground.   (https://link.springer.com/chapter/10.1007/978-3-030-01267-0_12  )  
  Deng-Ping Fan, Ming-Ming Cheng, Jiang-Jiang Liu, Shang-Hua Gao, Qibin Hou, Ali Borji  

   [Li-etal2018] Contour Knowledge Transfer for Salient Object Detection.   (https://link.springer.com/chapter/10.1007/978-3-030-01267-0_22  )  
  Xin Li, Fan Yang, Hong Cheng, Wei Liu, Dinggang Shen  

   [Fan-etal2018] Associating Inter-image Salient Instances for Weakly Supervised Semantic Segmentation.   (https://link.springer.com/chapter/10.1007/978-3-030-01240-3_23  )  
  Ruochen Fan, Qibin Hou, Ming-Ming Cheng, Gang Yu, Ralph R. Martin, Shi-Min Hu  

   [Chen-etal2018] Reverse Attention for Salient Object Detection.   (https://link.springer.com/chapter/10.1007/978-3-030-01240-3_15  )  
  Shuhan Chen, Xiuli Tan, Ben Wang, Xuelong Hu  

<h3><a id="user-content-2019-cvpr" class="anchor" aria-hidden="true" href="#2019-iccv"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2019 ICCV</h3>

   [Fan-etal2019] Understanding Human Gaze Communication by Spatio-Temporal Graph Reasoning. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Fan_Understanding_Human_Gaze_Communication_by_Spatio-Temporal_Graph_Reasoning_ICCV_2019_paper.pdf">[PDF]  
  Lifeng Fan, Wenguan Wang, Siyuan Huang, Xinyu Tang, Song-Chun Zhu   

   [Kellnhofer-etal2019] Gaze360: Physically Unconstrained Gaze Estimation in the Wild. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Kellnhofer_Gaze360_Physically_Unconstrained_Gaze_Estimation_in_the_Wild_ICCV_2019_paper.pdf">[PDF]  
  Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Matusik, Antonio Torralba   

   [He-etal2019] Photo-Realistic Monocular Gaze Redirection Using Generative Adversarial Networks. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/He_Photo-Realistic_Monocular_Gaze_Redirection_Using_Generative_Adversarial_Networks_ICCV_2019_paper.pdf">[PDF]  
  Zhe He, Adrian Spurr, Xucong Zhang, Otmar Hilliges   

   [Park-etal2019] Few-Shot Adaptive Gaze Estimation. <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf">[PDF]  
  Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges, Jan Kautz   
<h3><a id="user-content-2019-cvpr" class="anchor" aria-hidden="true" href="#2019-cvpr"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2019 CVPR</h3>

   [Xiong-etal2019] Mixed Effects Neural Networks (MeNets) With Applications to Gaze Estimation. <a href="">[PDF]  
  Yunyang Xiong, Hyunwoo J. Kim, Vikas Singh  

   [Yu-etal2019] Improving Few-Shot User-Specific Gaze Adaptation via Gaze Redirection Synthesis. <a href="http://www.idiap.ch/~odobez/publications/YuLiuOdobez-CVPR2019.pdf">[PDF]  
  Yu Yu, Gang Liu, Jean-Marc Odobez  

   [Wang-etal2019] Neuro-Inspired Eye Tracking With Eye Movement Dynamics. <a href="http://homepages.rpi.edu/~wangk10/papers/wang2019neural.pdf">[PDF]  
  Kang Wang, Hui Su, Qiang Ji  

   [Wang-etal2019] Generalizing Eye Tracking With Bayesian Adversarial Learning. <a href="https://www.semanticscholar.org/paper/Generalizing-Eye-Tracking-with-Bayesian-Adversarial-Wang-Zhao/77b9b6786699a236aad0c3fa3734730ece4a780f">[PDF]  
  Kang Wang, Rui Zhao, Hui Su, Qiang Ji  

<h3><a id="user-content-2019-others" class="anchor" aria-hidden="true" href="#2019-others"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2019 Others</h3>

   [Hickson-etal2019 WACV] Eyemotion: Classifying Facial Expressions in VR Using Eye-Tracking Cameras. <a href="https://doi.org/10.1109/WACV.2019.00178">[PDF]  
  Steven Hickson, Nick Dufour, Avneesh Sud, Vivek Kwatra, Irfan A. Essa  

   [Lian-etal2019 AAAI] RGBD Based Gaze Estimation via Multi-Task CNN.<a href="https://doi.org/10.1609/aaai.v33i01.33012488">[PDF]  
  Dongze Lian, Ziheng Zhang, Weixin Luo, Lina Hu, Minye Wu, Zechao Li, Jingyi Yu, Shenghua Gao  


<h3><a id="user-content-2018-eccv" class="anchor" aria-hidden="true" href="#2018-eccv"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.0  2018 ECCV</h3>

   [Fischer-etal2018] RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. <a href="https://link.springer.com/chapter/10.1007%2F978-3-030-01249-6_21" rel="nofollow">[PDF]  
  Tobias Fischer, Hyung Jin Chang, Yiannis Demiris  

   [Yohanandan-etal2018] Saliency Preservation in Low-Resolution Grayscale Images. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01231-1_15">[PDF]  
  Shivanthan Yohanandan, Andy Song, Adrian G. Dyer, Dacheng Tao  

   [Jiang-etal2018] DeepVS: A Deep Learning Based Video Saliency Prediction Approach. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01264-9_37">[PDF]  
  Lai Jiang, Mai Xu, Tie Liu, Minglang Qiao, Zulin Wang  

   [Kummerer-etal2018] Saliency Benchmarking Made Easy: Separating Models, Maps and Metrics. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01270-0_47">[PDF]  
  Matthias Kummerer, Thomas S. A. Wallis, Matthias Bethge  

   [Zheng-etal2018] Task-driven Webpage Saliency. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01264-9_18">[PDF]  
  Quanlong Zheng, Jianbo Jiao, Ying Cao, Rynson W.H. Lau  

   [Zhang-etal2018] Saliency Detection in 360° Videos. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01234-2_30">[PDF]  
  Ziheng Zhang, Yanyu Xu, Jingyi Yu, Shenghua Gao  

   [Song-etal2018] Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01252-6_44" rel="nofollow">[PDF]  
  Hongmei Song, Wenguan Wang, Sanyuan Zhao, Jianbing Shen, Kin-Man Lam  

   [Fan-etal2018] Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01267-0_12" rel="nofollow">[PDF]  
  Deng-Ping Fan, Ming-Ming Cheng, Jiang-Jiang Liu, Shang-Hua Gao, Qibin Hou, Ali Borji  

   [Li-etal2018] Contour Knowledge Transfer for Salient Object Detection. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01267-0_22" rel="nofollow">[PDF]  
  Xin Li, Fan Yang, Hong Cheng, Wei Liu, Dinggang Shen  

   [Fan-etal2018] Associating Inter-image Salient Instances for Weakly Supervised Semantic Segmentation. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01240-3_23" rel="nofollow">[PDF]  
  Ruochen Fan, Qibin Hou, Ming-Ming Cheng, Gang Yu, Ralph R. Martin, Shi-Min Hu  

   [Chen-etal2018] Reverse Attention for Salient Object Detection. <a href="https://link.springer.com/chapter/10.1007/978-3-030-01240-3_15" rel="nofollow">[PDF]  
  Shuhan Chen, Xiuli Tan, Ben Wang, Xuelong Hu  






<!---If you find the survey useful for your research, please consider citing our work:
```
@inproceedings{ghosh2018automatic,
  title={Automatic Group Affect Analysis in Images via Visual Attribute and Feature Networks},
  author={Ghosh, Shreya and Dhall, Abhinav and Sebe, Nicu},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={1967--1971},
  year={2018},
  organization={IEEE}
}-->
