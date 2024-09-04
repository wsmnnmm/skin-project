// pages/data/data.js
Page({

  /**
   * 页面的初始数据
   */
  data: {
    imageSrc:"/image/cancer1.jpg",
    textContent: '光化性角化病是一种职业病，主要受日光、紫外线、放射性热能以及沥青或煤及其提炼而物诱发本病。病损多见于中年以上男性日光暴露部位，如面部、耳廓、手背等。主要表现为表面粗糙，可见角化性鳞屑。揭去鳞屑，可见下方的基面红润，凹凸不平，呈乳头状。治疗一般采取外用药和手术治疗。有20%可继发鳞癌。',
    textContent1:'光化性角化病'
  },
  changeContent: function(e) {  
    const id = e.currentTarget.dataset.id; // 获取点击的按钮对应的id  
    switch(id) {  
      case '1':  
        this.setData({  
          imageSrc: '/image/cancer1.jpg', // 替换为你的图片URL  
          textContent: '光化性角化病是一种职业病，主要受日光、紫外线、放射性热能以及沥青或煤及其提炼而物诱发本病。病损多见于中年以上男性日光暴露部位，如面部、耳廓、手背等。主要表现为表面粗糙，可见角化性鳞屑。揭去鳞屑，可见下方的基面红润，凹凸不平，呈乳头状。治疗一般采取外用药和手术治疗。有20%可继发鳞癌。'  ,
          textContent1:'光化性角化病'
        });  
        break;  
      case '2':  
        this.setData({  
          imageSrc: '/image/cancer2.jpg', // 替换为你的图片URL  
          textContent: '基底细胞癌发生转移率低，比较偏向于良性，故又称基底细胞上皮瘤。基于它有较大的破坏性，又称侵袭性溃疡。基底细胞癌多见于老年人，好发于头、面、颈及手背等处，尤其是面部较突出的部位。开始是一个皮肤色到暗褐色浸润的小结节，较典型者为蜡样、半透明状结节，有高起卷曲的边缘。中央开始破溃，结黑色坏死性痂，中心坏死向深部组织扩展蔓延，呈大片状侵袭性坏死，可以深达软组织和骨组织。'  ,
          textContent1:'基底细胞癌'
        });break;  
        case '3':  
        this.setData({  
          imageSrc: '/image/cancer3.jpg', // 替换为你的图片URL  
          textContent: '角化病的病因是角化毛孔被角栓闭塞，呈毛孔性角化小丘疹，病因不明。部分病人有甲状腺机能低下，或有Cushing’’sSyndrome（库辛氏症候群）。也有一部分的病人是因为注射或服用皮质类固醇以后，才发生此种皮肤病。常见于异位性倾向的病人，或遗传性，多发于同一家族。所以，还没有开始医治时，需要作进一步的检查，诊断毛囊角化病是属于哪一种。治疗的药物包括维甲酸霜剂或凝胶。AHA、PHA、KogicAcid或是阿达帕林凝胶。这些霜剂或凝胶大多会有一些不良的反应。'  ,
          textContent1:'良性角化样病变'
        }); break;  
        case '4':  
        this.setData({  
          imageSrc: '/image/cancer4.jpg', // 替换为你的图片URL  
          textContent: '皮肤纤维瘤是成纤维细胞或组织细胞灶性增生引致的一种真皮内的良性肿瘤。本病可发生于任何年龄，中青年多见，女性多于男性。可自然发生或外伤后引起。黄褐色或淡红色的皮内丘疹或结节是本病的临床特征。病损生长缓慢，长期存在，极少自行消退。本病的真正病因不明。' ,
          textContent1:'皮肤纤维瘤' 
        });break;  
        case '5':  
        this.setData({  
          imageSrc: '/image/cancer5.jpg', // 替换为你的图片URL  
          textContent: '黑色素瘤，通常是指恶性黑色素瘤，是黑色素细胞来源的一种高度恶性的肿瘤，简称恶黑，多发生于皮肤，也可见于黏膜和内脏，约占全部肿瘤的3%。皮肤恶性黑色素瘤占皮肤恶性肿瘤的第三位（约占6.8%~20%）,近年来，恶性黑色素瘤的发生率和死亡率逐年升高，与其他实体瘤相比，其致死年龄更低。恶性黑色素瘤除早期手术切除外，缺乏特效治疗，预后差。因此，恶性黑色素瘤的早期诊断和治疗极其重要。'  ,
          textContent1:'黑色素瘤'
        });break;  
        case '6':  
        this.setData({  
          imageSrc: '/image/cancer6.jpg', // 替换为你的图片URL  
          textContent: '血管性皮肤病（vascular dermatosis）是指原发于皮肤血管管壁的一类炎症性疾病，其共同组织病理表现为血管内皮细胞肿胀，血管壁纤维蛋白样变性及管周炎症细胞浸润或肉芽肿形成。',
          textContent1:'血管性皮肤病变'  
        });break;  
        case '7':  
        this.setData({  
          imageSrc: '/image/cancer7.jpg', // 替换为你的图片URL  
          textContent: '又称痣细胞痣（Nevocytic nevus），是一种人类常见的良性肿瘤，发生于皮肤的黑素细胞（痣细胞）。黑素细胞痣的天生携带率达到1%，常在2岁后发生。而后天也可能由于紫外线照射等原因产生，一般为良性。'  ,
          textContent1:'黑素细胞痣'
        }); 
        break;  
      // 根据需要添加更多case  
    }  
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {

  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})