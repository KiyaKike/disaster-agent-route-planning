"""Independent strip and interpolation checks;close data-admission decision."""
import json,hashlib,shutil,math
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile
from check_noaa_verification_entry import ROOT,O,P,S
from affine import Affine
from pyproj import Transformer
from crasar_alignment_core import nearest_shift_indices

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def run():
    if (O/'completion_receipt.json').exists():raise RuntimeError('Sealed')
    result=read(O/'entry_assessment.json');acq=read(O/'acquisition.json');meta=read(O/'source_metadata.json');roi=read(P/'roi_protocol.json')
    assert acq['cached_bytes']<read(O/'protocol.json')['payload_cap_bytes']
    files={int(f.stem):f.read_bytes() for f in (O/'ranges').glob('*.bin')}
    def bytes_at(start,n):
        out=[]
        while n:
            k,offset=divmod(start,65536);assert k in files
            take=min(n,len(files[k])-offset);assert take>0
            out.append(files[k][offset:offset+take]);start+=take;n-=take
        return b''.join(out)
    native=tuple(meta['native_window']);x0,y0,w,h=native
    # Decode raw uncompressed strip bytes independently of GDAL and the sparse container.
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(O/'crewed_partial.sparse.tif') as ds:raw=np.moveaxis(ds.read([1,2,3],window=Window(*native)),0,-1)
    rng=np.random.default_rng(20261103);sampled=rng.integers([0,0],[h,w],size=(100,2))
    with tifffile.TiffFile(O/'crewed_partial.sparse.tif') as tf:
        page=tf.pages[0];assert page.compression==1 and page.planarconfig==1 and page.rowsperstrip==1
        for rr,cc in sampled:
            offset=page.dataoffsets[int(rr)+y0]+(int(cc)+x0)*3
            assert bytes_at(offset,3)==raw[rr,cc].tobytes()
        for strip in range(y0,y0+h):
            a=page.dataoffsets[strip];n=page.databytecounts[strip]
            assert set(range(a//65536,(a+n-1)//65536+1))<=set(files)
    # Compare independently computed bilinear samples to fixed OpenCV table quantization.
    inverse=~Affine(*meta['transform'][:6]);tx=Transformer.from_crs(roi['crs'],meta['crs'],always_xy=True)
    vectors=np.array(read(O/'crewed_alignment.json'));shifts=vectors[:,1]-vectors[:,0]
    points=rng.integers(0,1024,size=(500,2));rr,cc=points[:,0],points[:,1]
    gx=roi['bounds'][0]+(cc+.5)*.125;gy=roi['bounds'][3]-(rr+.5)*.125
    sx,sy=tx.transform(gx,gy);qx,qy=inverse*(sx,sy)
    idx=nearest_shift_indices(vectors[:,0],qx,qy)
    maxerror={}
    for mode,px,py in [('raw_georef',qx,qy),('aligned',qx+shifts[idx,0],qy+shifts[idx,1])]:
        x=px-x0-.5;y=py-y0-.5;ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);fx=(x-ix)[:,None];fy=(y-iy)[:,None]
        expected=(1-fx)*(1-fy)*raw[iy,ix]+fx*(1-fy)*raw[iy,ix+1]+(1-fx)*fy*raw[iy+1,ix]+fx*fy*raw[iy+1,ix+1]
        actual=np.array(Image.open(O/f'crewed_{mode}_roi.png'))[rr,cc]
        err=float(abs(expected-actual).max());assert err<=9
        maxerror[mode]=err
    # Recompute support and disagreements from original published records,without a model.
    records=read(O/'private_view_reference_comparison.json');known={'no damage','minor damage','major damage','destroyed'}
    eligible=[r for r in records if r['existing_geometry_eligible'] and r['crewed_id_exists'] and r['crewed_image_support_fraction']>=.9]
    assert len(eligible)==12 and result['shared_ids_with_roi']==16
    for mode in ['satellite','uas']:
        pairs=[r for r in eligible if r['view_labels'][mode] in known and r['view_labels']['crewed'] in known]
        expected=sum(r['view_labels'][mode]!=r['view_labels']['crewed'] for r in pairs)
        assert expected==result['view_label_comparison'][mode]['disagreements']
    previous={}
    for name in ['disaster_route_crasar_region_rank2_v1','disaster_route_verification_screen_v1','disaster_route_research_admission_v1','disaster_route_crasar_batch4_v1']:
        folder=ROOT/'output'/name;receipt=read(folder/'completion_receipt.json')
        for path,h in receipt['files'].items():assert sha(folder/path)==h,(name,path)
        previous[name]=len(receipt['files'])
    audit={'status':'COMPLETE_ENTRY_AUDIT_WITH_NEGATIVE_CALIBRATION_ADMISSION','native_raw_RGB_pixels_independently_verified':100,
      'bilinear_points_per_variant':500,'bilinear_variants':2,'max_abs_interpolation_error_8bit':maxerror,
      'tolerance_8bit':9,'tolerance_reason':'OpenCV1/32pixel linear-interpolation table plus rounding;independent exact bilinear comparison,not a registration tolerance',
      'all_required_strips_present':True,'decoded_strip_rows':native[3],'crewed_supported_existing_eligible':12,
      'archive_image_access_supported':True,'calibration_anchor_validated':False,'same_state_UAS_reinspection_validated':False,
      'causal_error_attribution_validated':False,'method_gain_claim':False,'new_model_calls':0,'new_training':False,'gpu_rental':False,
      'previous_frozen_files_verified':previous}
    dump(O/'audit.json',audit)
    decision={'archive_cross_source_entry':'PIXEL_SUPPORT_CONFIRMED_ONLY','unknown_error_attribution_entry':'NOT_QUALIFIED',
      'calibration_anchor':'NOT_ESTABLISHED','main_candidate_status':'STOPPED_AND_NOT_REOPENED',
      'next_step':'Complete bounded admission review;do not add inference or route runs on this entry. Discuss new verifiable research scope before another algorithm candidate.',
      'needed_to_reopen_physical_verification':['Same-building independent acquisitions with verified acquisition times/state assumptions','Common physical reference or defensible calibrated observation model','Specific method distinction beyond generic Bayesian/robust information value'],
      'alternative_scope':'Archived cross-source evidence access under explicitly simulated motion costs;no automatic novelty or real-flight claim'}
    dump(O/'decision.json',decision)
    text=f'''# NOAA复核入口：实际影像与研究可用性审查

2026-11-03。**实际图像入口成立；作为“可靠校准锚点／同状态无人机复访”的入口尚未成立。** 这项结果不改变上一轮停止泛化复核候选的决定。

沿用此前按元数据排序确定的Boone、原128m区域及建筑集合，不因标签或方法结果换例。实际读取NOAA三通道GeoTIFF的533行源条带，缓存{acq['cached_bytes']:,}字节（原文件{acq['logical_file_bytes']:,}字节）；未下载完整影像。全部原16栋区域内建筑中心都有NOAA对应ID，原12栋完整可用建筑在NOAA影像中也全部有非黑像素支持。没有新增模型推理或训练。

![同一区域跨源影像与局部修正](entry_panel.png)

## 核验发现

|项目|结果|可支持的结论|
|---|---|---|
|影像支持|原12栋可用建筑支持比例均100%|这里确有可读取的额外图像；不等于无遮挡或能判清损伤|
|作者目录日期|灾前卫星2022-01-21；灾后卫星和NOAA均09-30；UAS10-01|有跨源时间关系；没有精确拍摄时刻、目标状态稳定性证明或飞行记录|
|NOAA／卫星视角标签|12栋已知标签中2栋不同|存在标注／可见证据差异，不能指定哪一方正确|
|NOAA／UAS视角标签|12栋已知标签中1栋不同|仍不能分离真实变化、遮挡、配准、判读差异与检测错误|
|共同现场真值与校准模型|未提供／未建立|不能将NOAA当成无误差教师，不能直接估计可靠漏检率|

标签对照是作者影像视角标注，不是新检测器输出：卫星4栋minor中，NOAA有2栋minor、2栋no damage；UAS3栋minor中，NOAA有2栋minor、1栋no damage。没有把多数意见当真值，也没有用这些标签作为免费Agent输入。该小区域不含严重损伤目标，不能评价严重搜查收益。

## 配准结果不宜简化成“已通过”

保留原地理变换和作者局部修正两种NOAA显示，不按损伤结果或匹配得分选择版本。固定SIFT诊断中，卫星到NOAA中位直接残差从约1.28m变为2.02m（分别42／35个平移一致匹配）；UAS到NOAA从约4.43m变为1.61m（8／6个一致匹配）。灾前卫星到NOAA未得到足够一致匹配。

这些是局部自动匹配诊断，不是独立地面控制点或测绘精度。作者局部修正并未对两组关系一致改善，不能把它当成自动通过配准门槛的证书。保留两版本不表示补造第二次观测。

## 研究结论与止损

这个来源可以进入**档案跨源证据访问**的候选资料，但还不能验证“先验错误还是检测器漏检”的因果归因，更不能充当上一轮小模型里返回世界身份的校准信号Q。独立取得的图像也不自动具有独立误差。

作者的[跨源损伤标注审计](https://arxiv.org/abs/2505.08117)已指出卫星／无人机标签不一致的现实问题；这不是我们新发现的研究空白。其[实际部署记录](https://arxiv.org/abs/2506.15890)也明确列出分辨率、配准和连接等问题，不能靠再添加一张正射影像就视为消除这些条件。

**本轮既定数据入口检查到此结束，不新增路线运行或感知训练来延长已停止候选。** 当前仍没有可主张的新方法贡献。恢复物理复核方向，需要同建筑、时间相容的独立采集，能够支持共同损伤参考或观测模型校准，并且先明确区别于已有方法的新命题。资源不足时，应与老师讨论把实证范围明确收缩到档案证据访问，或更换候选；不能把该收缩本身写成创新。

检查证据：100个源像素用缓存中的未压缩条带字节独立核对，1000个插值样本与独立双线性计算核对，全部所需条带存在；旧{sum(previous.values())}个封存文件哈希不变。插值容差仅用于8位像素数值验证，不是空间配准容差。详见[audit.json](audit.json)、[决策记录](decision.json)及[来源元数据](source_metadata.json)。
'''
    (O/'README.md').write_text(text,encoding='utf-8')
    snapshots=O/'source_snapshots';snapshots.mkdir(exist_ok=True)
    for name in ['check_noaa_verification_entry.py','finish_noaa_verification_entry.py','probe_crasar_region.py','crasar_alignment_core.py','qualify_crasar_region.py']:
        shutil.copyfile(ROOT/'work/disaster_route'/name,snapshots/name)
    files={str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file() and '.sparse.tif' not in p.name and p.name!='completion_receipt.json'}
    dump(O/'completion_receipt.json',{'status':'COMPLETE_ARCHIVE_ENTRY_DIAGNOSTIC_NOT_CALIBRATION_VALIDATION','files':files,
      'excluded_ephemeral':'Partial sparse TIFF is not a full source image;source byte caches and derived ROIs are sealed',
      'new_method_claim':False,'calibration_anchor_validated':False})
    print(json.dumps({'audit':audit,'sealed_files':len(files),'decision':decision},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
