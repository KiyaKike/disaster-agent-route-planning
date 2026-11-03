"""Reset the public cell inspector when changing region or route policy.

Browser testing found stale text after the selected-cell index was cleared.
This changes the generated presentation only;frozen agent sources/data stay intact.
"""
from build_dual_map_demo import OUT,read,save,digest


def main():
    assert not (OUT/'completion_receipt.json').exists()
    path=OUT/'demo.html';before=digest(path);text=path.read_text()
    old='function load(){stop();selected=null;const {g,t}=currentSafe();'
    new="function load(){stop();selected=null;$('inspector').textContent='点击地图格子，查看线索、当时后验和已付费证据。黑色实线为已走路径，绿色虚线为当前剩余计划；虚线包含返程。';const {g,t}=currentSafe();"
    assert text.count(old)==1
    path.write_text(text.replace(old,new,1))
    save(OUT/'viewer_amendment_01.json',{'scope':'Presentation only;clear stale cell inspector on region/policy changes.',
        'agent_sources_or_route_data_changed':False,'original_html_sha256':before,'amended_html_sha256':digest(path),
        'postprocessor_sha256':digest(__file__)})
    print('VIEWER_INSPECTOR_RESET_FIXED')


if __name__=='__main__':main()
