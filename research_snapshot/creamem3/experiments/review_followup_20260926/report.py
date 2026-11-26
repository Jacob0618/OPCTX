"""Consolidated favorable/unfavorable evidence, no manuscript edits."""
from pathlib import Path
from . import run as r
from creamem3.experiments.review_fast_20260926 import audit as a

def ci(x):
    lo,hi=x['cluster_bootstrap95_pp'];return f"{x['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"

def main():
    result=r.b.read(sorted((r.ROOT/'analysis').glob('*.json'))[-1])
    assert r.b.read(r.ROOT/'acceptance.json')['status']=='PASS'
    loc=r.b.read(r.ROOT/'locomo_evidence/results.json');pert=r.b.read(r.ROOT/'locomo_view_perturbation/results.json')['summary']
    solve=r.b.read(r.ROOT/'solver_sensitivity/summary.json');judges=r.b.read(r.ROOT/'judge_robustness/results.json')
    names={'bm25':'BM25 Top30','fusion':'等权多通道融合 Top30','mmr':'MMR Top30（λ=0.5）','generic_reasoning':'相同数学30＋通用推理Reader'}
    lines=['# 审稿补实验第二轮：哪些支持、哪些削弱（0926）','',
        '本轮四组MemEye全量371题已经完成，另补LoCoMo全量证据来源诊断、20种随机事件连边负对照、求解时限敏感性及已有双Judge一致性分析。全部结果保留，不按方向筛选；论文及旧实验文件未改。','',
        '## 1. 最直接的结论','',
        '- 支持Reader整体作用：固定facts时，Reader相对直接QA的增益在两个Judge下均成立。但本轮通用推理Reader46.23%，与Full47.98%的差值区间跨零，尚不能把整体收益归因于操作专用schema。',
        '- 支持LoCoMo事件连接的检索作用：相同Writer事实库，Online提高候选及最终facts对标注来源的覆盖；打乱事件连接后，多跳来源召回明显下降。',
        '- 明显削弱“当前数学目标更优”的说法：上轮同池同30条的数学与Dense均47.98%；本轮等权多通道融合52.16%，高于数学Full47.98%。Full优于MMR40.97%，但这不能抵消融合对照的结果。',
        '- 原MemEye38.54/37.47两组输入完全相同，不能称Online消融；LoCoMo Reader增量0.65个百分点的区间也跨零，不能声称每个数据集每个组件都有稳定提升。',
        '- 现有成本可以支持免rerank、受控读取和逐阶段资源描述，尚不支持比所有baseline端到端更便宜。','',
        '## 2. 本轮全量MemEye结果','',
        '复用现有48.92实验的冻结记忆。生成3.5 Flash-Lite，Judge 3.5 Flash，无rerank；参考为上一轮同期复测47.98%，不挑选最高复测。','',
        '| 配置 | 分数 (%) | Full−该对照，百分点 [场景聚类95%CI] |','|---|---:|---|',
        f"| Full参考 | {result['full_reference_percent']:.2f} | — |",
        '| 上轮同池Dense30 | 47.98 | +0.00 [-3.91, +4.88] |']
    for arm in r.ARMS:lines.append(f"| {names[arm]} | {result['scores'][arm]:.2f} | {ci(result['comparisons'][arm])} |")
    lines += ['',
        '选择器三组都使用全部可见Writer facts、同一字符预算b(q)、最多30条、同一最终dense展示顺序、原增强Reader和同一QA。融合采用原有活跃百分位效用通道的等权平均，保留原问题与冻结查询投影；MMR固定λ=0.5，没有调参。BM25零分及同分使用dense和ID排序。',
        '通用推理Reader固定Full的facts及顺序，允许普通证据说明、简短推理和候选答案，只不使用问题分类及计数/时间/比较专用结构。306题仍两次调用，65题仍融合式一次调用；两阶段的独立QA提示词不变。',
        '提示词给出仅由历史输出token推算的近似篇幅目标，未用正确率调长度。输出上限及调用次数匹配，实际token仍须报告，不能说精确等计算。','',
        '| 配置 | 调用数（不含Judge） | 输入token/题 | 输出token/题 |','|---|---:|---:|---:|']
    for arm in ['full_refresh',*r.ARMS]:
        usage=result['costs'][arm];lines.append(f"| {names.get(arm,'Full参考')} | {sum(v['calls'] for v in usage.values())} | {sum(v['prompt_tokens'] for v in usage.values())/371:.1f} | {sum(v['output_tokens'] for v in usage.values())/371:.1f} |")
    total=lambda arm:sum(s['prompt_tokens']+s['output_tokens'] for s in result['costs'][arm].values())
    lines += ['',f"通用推理对照的实际输入＋输出token比Full少{100*(1-total('generic_reasoning')/total('full_refresh')):.1f}%，调用次数均677。Full−通用推理差值区间跨零，不能继续把前一轮摘要对照的19.68点全部归因于operation-conditioned schema。",'',
        f"通用推理对照两阶段306题：Full−对照 {ci(result['generic_route_details']['two_stage'])}。融合路径只有2个场景，不将其区间单独作为强证据。",'',
        '融合高于Full的普通bootstrap区间不跨零，但8场景精确sign-flip经4项Holm校正后未达到0.05；报告为不利的实测结果，不把一次比较夸大成普遍优劣。Full对MMR的校正p为0.03125。', '',
        '额外代理目标检查显示：融合只有149/371题满足Full的字面覆盖最优域，平均少覆盖1.57个问题词，却获得更高QA。它提示覆盖/均衡代理与答案效用可能不一致，不能仅凭这组结果判断是coverage还是minimax造成。融合若不在同一覆盖域，其utility比率不能套用原定理。下一步更有价值的对照是固定通道与预算，只改变“是否硬锁覆盖”和“均值还是minimax”，不按本371题结果反复调权重。','',
        '上述区间按8个完整场景配对重采样10,000次，seed92602，按题数加权；仍是开发过的冻结数据，不等同独立held-out验证。4项固定比较的探索性Holm校正结果保存在JSON中。','',
        '## 3. LoCoMo：Online究竟改变了什么','',
        '对应历史2＋5＋5→rerank5、79.03/78.38/76.49配置，不是81.75主表配置。全量1540题中1536题有可解析的gold对话来源。两个模式Writer事实完全相同。','',
        '| 指标 | Session-only | Online |','|---|---:|---:|']
    overall=loc['summary']['overall'];multi=loc['summary']['by_category']['multi']
    source_ci=r.b.read(r.ROOT/'locomo_evidence/paired_source_recall.json')
    for label,part,stage in [('全部题候选池来源召回',overall,'candidate'),('全部题最终5facts来源召回',overall,'selected'),('多跳题候选池来源召回',multi,'candidate'),('多跳题最终5facts来源召回',multi,'selected')]:
        lines.append(f"| {label} | {part['session_strong'][stage]['mean_source_recall']*100:.2f}% | {part['opctx_original'][stage]['mean_source_recall']*100:.2f}% |")
    lines += [f"| 多跳QA | {multi['session_percent']:.2f}% | {multi['online_percent']:.2f}% |",'',
        f"多跳候选来源召回增益及对话聚类95%区间为 {ci(source_ci['candidate_multi'])}，最终facts来源召回为 {ci(source_ci['selected_multi'])}。",'',
        '1273题的top-summary实际命中event view。895题最终facts集合改变；其余645题集合相同。使用普通facts呈现的Online控制是78.38%，相比Session-only76.49仍有增益，说明差别不只来自把view正文交给QA。',
        '这里的来源召回是：选中facts引用的对话ID覆盖多少标注evidence ID。它不保证facts完整保留原句语义，也没有把summary/view正文中的额外答案算成facts召回。','',
        '### 随机事件连接负对照','',
        '20个预先固定随机种子，按对话对fact ID作全局置换，仅替换event-view连边。每个view的成员数与view间交叠结构保持，session连边、summary排序、dense/BM25分支及120候选上限不变。先逐题复现原候选顺序，1540/1540一致，再计算扰动。','',
        f"- 全部题候选来源召回：原连接 {pert['all']['original_recall_percent']:.2f}%，随机连接平均 {pert['all']['shuffled_mean_percent']:.2f}%；配对差值 {ci(pert['paired_source_recall'])}。",
        f"- 多跳题：原连接 {pert['by_category']['multi']['original_recall_percent']:.2f}%，随机连接平均 {pert['by_category']['multi']['shuffled_mean_percent']:.2f}%。",
        '- 这支持已有连边具有检索信息，不是随机扩大候选就有同等效果。实验止于候选阶段，没有重跑随机图的rerank/QA，也不提供人工事件precision。','',
        '## 4. 求解器与Judge稳健性','',
        f"- 原56道不完整证书题，将每阶段时限从5秒加到30秒后，{solve['same_ordered']}/56题选中ID及顺序完全相同，平均重合30/30；1题新增完整证书，其余仍须按数值核验结果标为partial，不能因延长时限冒称已证明最优。",
        '- 因为这些题上下文完全不变，没有额外花钱重做QA来制造“求解器敏感性”；结论是选择对延长时间稳定。','',
        '| 原Reader−直接QA（同一facts） | 增益及95%区间 |','|---|---|']
    for j,x in judges['original_reader_minus_direct'].items():lines.append(f'| {j} | {ci(x)} |')
    j=judges['summaries']['enhanced_reader']['judge_delta']
    lines += [f"\n同一Full答案由两种Judge评分为{j['left_percent']:.2f}%与{j['right_percent']:.2f}%，差值{ci(j)}；这属于Judge模型敏感性，不是人工正确率。",'',
        '## 5. 论文该怎么用这些证据','',
        '1. 正文优先放同池选择器对照、强通用Reader对照，以及LoCoMo多跳来源召回。它们分别回答“选择规则是否有效”“是否只是多一次LLM调用”“事件连接在哪一步起作用”。',
        '2. 附录放聚类区间、Judge差异、求解时限检查、随机连接分布和逐阶段成本。不把不同模型、不同K和不同Reader配置的分数混成一条递减消融。',
        '3. 叙述可以保留“事件组织改善跨会话证据访问，Reader把证据转成可回答的信息”。不能继续说现有minimax选择比简单融合更准确，也不能把Reader整体作用直接等同于专用schema创新已被证明。',
        '4. 还缺事件view的真实人工语义审核、纠错/冲突专项测试、新近可复现baseline以及完整baseline端到端成本。本轮没有把AI初筛称成人工审核，也没有声称解决了这些缺口。','',
        '## 6. 文件与复现','',
        '- 新结果：`D:/opctx_review26/r2/`；原参考：`D:/opctx_review26/r1/`。',
        '- 代码：`C:/Users/86139/Desktop/agent memory/creamem3/experiments/review_followup_20260926/`。',
        '- 主运行、协议、输入及请求哈希均保留；原论文、历史结果和缓存未删除或覆盖。',
        '- 个别Judge响应的未转义引号仅作保留原score和reasoning的无损修复，原响应与故障证据保留，不重新生成或择优。']
    text='\n'.join(lines)+'\n'
    for path in [r.ROOT/'report.md',Path(r'C:\Users\86139\Desktop\审稿补实验_第二轮_0926.md')]:
        path.write_text(text,encoding='utf-8')
    r.b.save(r.ROOT/'report_provenance.json',dict(code_sha256=r.b.file_sha(__file__),report_sha256=r.b.sha(text)))
    print('REPORT READY')

if __name__=='__main__':main()
