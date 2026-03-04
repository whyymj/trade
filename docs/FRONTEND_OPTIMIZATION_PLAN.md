# 前端页面优化方案

## 目标
简化用户操作，将分散的多页面功能整合到 FundDetail 页面，减少跳转次数。

## 优化方案

### 1. FundDetail 新增"行业新闻"Tab (高优先级)

**问题**: 目前查看行业分析、相关新闻需要跳转到独立页面并重新输入基金代码

**解决方案**: 在 FundDetail 添加第6个 Tab

```
现有Tabs: [净值走势] [业绩指标] [基准对比] [LLM分析]
新增Tab:   [行业新闻]
```

**功能内容**:
- 行业配置（置信度条 + 行业标签）
- 相关新闻列表（5条）
- 情绪分析（积极/消极/中性 + 图标）
- 一键分析按钮（触发行业分析+新闻分类+关联匹配）

**API 调用**:
- `/api/fund-industry/:code` - 获取行业配置
- `/api/fund-news/summary/:code` - 获取相关新闻摘要

---

### 2. FundHome 基金卡片显示行业标签 (中优先级)

**问题**: 无法快速识别基金持仓方向

**解决方案**: 卡片添加行业标签

```vue
<!-- FundHome.vue 修改 -->
<div class="fund-card">
  <div class="fund-name">{{ fund.fund_name }}</div>
  <!-- 新增 -->
  <div class="industry-tags" v-if="fund.industries">
    <span v-for="ind in fund.industries.slice(0, 2)" :key="ind" class="tag-sm">
      {{ ind }}
    </span>
  </div>
</div>
```

**效果**: 
- 鼠标悬停显示完整行业列表
- 快速识别基金持仓方向

---

### 3. 参数透传优化 (中优先级)

**问题**: 页面跳转时数据丢失

**解决方案**: FundHome → FundDetail 传递必要数据

```javascript
// FundHome.vue
router.push({
  path: `/fund/${fund.fund_code}`,
  query: { 
    industries: JSON.stringify(fund.industries || [])
  }
})

// FundDetail.vue  
const industries = computed(() => {
  const q = route.query.industries
  return q ? JSON.parse(q) : null
})
```

---

### 4. NewsClassification 增强搜索 (低优先级)

**问题**: 无法按关键词搜索特定新闻

**解决方案**: 添加搜索框

```vue
<!-- NewsClassification.vue -->
<div class="search-box">
  <input v-model="searchKeyword" placeholder="搜索新闻..." />
  <button @click="searchNews">搜索</button>
</div>
```

---

## 实施计划

### Phase 1: FundDetail 行业新闻 Tab
- [ ] 修改 FundDetail.vue 添加 Tab
- [ ] 添加行业配置展示组件
- [ ] 添加相关新闻列表组件
- [ ] 添加情绪分析组件
- [ ] 添加一键分析功能

### Phase 2: FundHome 行业标签
- [ ] 修改 FundHome.vue 卡片样式
- [ ] 添加行业标签展示

### Phase 3: 参数透传
- [ ] 修改 FundHome 跳转逻辑
- [ ] 修改 FundDetail 接收参数逻辑

### Phase 4: NewsClassification 搜索
- [ ] 添加搜索框组件
- [ ] 添加搜索 API 调用

---

## 预期效果

| 优化前 | 优化后 |
|--------|--------|
| 查看行业分析 → 跳转+输入代码 | 基金详情页直接查看 |
| 查看相关新闻 → 跳转+输入代码 | 基金详情页直接查看 |
| 需要多次输入代码 | 一次点击完成所有操作 |

---

## 实施顺序

1. **Phase 1** - FundDetail 行业新闻 Tab（预计2小时）
2. **Phase 2** - FundHome 行业标签（预计30分钟）
3. **Phase 3** - 参数透传（预计30分钟）
4. **Phase 4** - NewsClassification 搜索（预计30分钟）
