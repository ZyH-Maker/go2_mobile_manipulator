"""Fixed terminal display only; never connects to ROS or CAN."""
import math
import shutil
import sys
import unicodedata

PHASE_NAMES={
    'monitor':'只读监视', 'reset_pending':'退出原模式', 'resetting':'复位等待',
    'enabling':'使能中', 'activating':'确认位置模式', 'recovering':'边界恢复',
    'active':'已接管', 'held':'位置保持', 'trajectory':'轨迹执行中', 'fault':'故障锁定',
}


def angle(values, index):
    try:
        value=float(values[index])
        return '%9.3f'%math.degrees(value) if math.isfinite(value) else '       --'
    except (TypeError,ValueError,IndexError,KeyError):return '       --'


def screen_lines(node, now):
    joint=node.mode=='joint'
    online=now-node.status_at<.2
    fresh=now-node.feedback_at<.1
    phase=PHASE_NAMES.get(node.status.get('phase'),'等待状态') if online else '等待执行节点'
    mode='Home 回零中' if node.homing else ('键盘暂停' if node.paused else '运动键已启用')
    lines=[
        'PiPER 机械臂键盘控制 — '+('单关节点动' if joint else '末端控制'),
        '阶段：%s  |  %s'%(phase,mode),
        '执行节点：%s  |  实际角度反馈：%s'%('在线' if online else '未连接/超时','正常' if fresh else '未收到/超时'),
    ]
    if joint:
        lines.append('选中：J%d  |  每次步长：%.1f°'%(node.generator.selected+1,math.degrees(node.generator.step_rad)))
    lines+=['', '关节      实际角度(°)    下发目标(°)']
    # target_position is the executor's interpolated command, not the
    # keyboard endpoint and never a substitute for measured feedback.
    target=node.status.get('target_position') if online else None
    for i in range(6):
        mark='>' if joint and node.generator.selected==i else ' '
        lines.append('%s J%d      %s      %s'%(mark,i+1,angle(node.actual,i),angle(target,i)))
    lines+=['', 'E/Enter 接管/继续；R 越界恢复（会运动）' if joint else 'Enter 接管/继续；R 越界恢复（会运动）']
    if joint:lines+=['1～6 选择关节；J 减小 / L 增大；[ 小步 / ] 大步']
    else:lines+=['A/D 底座；Q/E 前后；W/S 上下；J/L J5；U/O J6；B Home']
    lines+=['P 取消并保持；空格急停；X / Ctrl+C 退出', '',
            '提示：'+node.notice]
    if node.status.get('error'):lines.append('执行节点提示：'+str(node.status['error']))
    if not fresh and node.actual is not None:lines.append('表中实际角度为上次收到的值，当前反馈已过期。')
    return lines


def clip_line(text, width):
    """Clip by terminal columns, including Chinese full-width characters."""
    result=[];columns=0
    for char in text:
        if unicodedata.category(char).startswith('C'):continue
        size=0 if unicodedata.combining(char) else (2 if unicodedata.east_asian_width(char) in ('W','F') else 1)
        if columns+size>width:break
        result.append(char);columns+=size
    return ''.join(result)


def draw_screen(lines):
    width,height=shutil.get_terminal_size((100,28))
    # Some container/SSH pseudo-terminals initially report a zero-sized window.
    if width<=0:width=100
    if height<=0:height=28
    visible=[clip_line(line,max(1,width-1)) for line in lines[:max(1,height-1)]]
    sys.stdout.write('\x1b[H'+'\r\n'.join(line+'\x1b[K' for line in visible)+'\x1b[J')
    sys.stdout.flush()
