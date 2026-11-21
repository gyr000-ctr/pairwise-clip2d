"""clip2d 核心 API。

- ``Clipper.clip(poly, rect)``：矩形裁剪（Sutherland–Hodgman），O(n)。
- ``clip_polygon(subject, clip)`` / ``Clipper.clip_poly``：凹多边形对凹多边形
  相交裁剪（Weiler–Atherton 思路），返回多边形环列表，输出顶点序稳定。
- ``polygon_area`` / ``intersection_area`` / ``union_area``：面积工具，
  满足 union_area + intersection_area == area(a) + area(b)（按构造守恒）。

所有函数均为纯函数：不修改入参、不保存状态，并发调用同一输入结果一致。
退化输入（零面积多边形、非法矩形、非有限坐标）抛出 ``ValueError``。
"""

import math

__all__ = [
    "Clipper",
    "clip_polygon",
    "polygon_area",
    "intersection_area",
    "union_area",
]

_T_TOL = 1e-9  # 线段参数空间容差


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _as_float(v, what):
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isfinite(f):
            return f
    raise ValueError("%s 必须是有限数值，得到 %r" % (what, v))


def _validate_rect(rect):
    try:
        xmin, ymin, xmax, ymax = rect
    except (TypeError, ValueError):
        raise ValueError("rect 必须是 (xmin, ymin, xmax, ymax) 四元组") from None
    xmin = _as_float(xmin, "rect.xmin")
    ymin = _as_float(ymin, "rect.ymin")
    xmax = _as_float(xmax, "rect.xmax")
    ymax = _as_float(ymax, "rect.ymax")
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("非法矩形：需要 xmax > xmin 且 ymax > ymin")
    return xmin, ymin, xmax, ymax


def _validate_poly(poly):
    try:
        iterator = iter(poly)
    except TypeError:
        raise ValueError("多边形必须是顶点可迭代对象") from None
    pts = []
    for p in iterator:
        try:
            x, y = p
        except (TypeError, ValueError):
            raise ValueError("多边形顶点必须是 (x, y) 对，得到 %r" % (p,)) from None
        pts.append((_as_float(x, "顶点 x"), _as_float(y, "顶点 y")))
    return pts


# ---------------------------------------------------------------------------
# 基础几何
# ---------------------------------------------------------------------------

def _signed_area2(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def _dedupe(poly):
    out = []
    for p in poly:
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _point_in_poly(pt, poly):
    """射线法点在多边形内判定；边界上视为内部。"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        tol = 1e-9 * max(abs(dx), abs(dy), 1.0) ** 2
        cross = (x - x1) * dy - (y - y1) * dx
        if abs(cross) <= tol:
            d = (x - x1) * dx + (y - y1) * dy
            if -tol <= d <= dx * dx + dy * dy + tol:
                return True
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def _seg_seg(p1, p2, p3, p4, dist_tol):
    """线段求交，返回 [(t, u), ...]（含端点接触与共线重叠端点）。"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    # 包围盒快速剔除
    if (x1 < x3 and x1 < x4 and x2 < x3 and x2 < x4) or \
            (x1 > x3 and x1 > x4 and x2 > x3 and x2 > x4) or \
            (y1 < y3 and y1 < y4 and y2 < y3 and y2 < y4) or \
            (y1 > y3 and y1 > y4 and y2 > y3 and y2 > y4):
        return []
    dx = x2 - x1
    dy = y2 - y1
    ex = x4 - x3
    ey = y4 - y3
    denom = dx * ey - dy * ex
    if denom != 0.0:
        t = ((x3 - x1) * ey - (y3 - y1) * ex) / denom
        if t < -_T_TOL or t > 1.0 + _T_TOL:
            return []
        u = ((x3 - x1) * dy - (y3 - y1) * dx) / denom
        if u < -_T_TOL or u > 1.0 + _T_TOL:
            return []
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
        return [(t, u)]
    # 平行：检查是否共线（点到直线距离容差）
    if (dx == 0.0 and dy == 0.0) or (ex == 0.0 and ey == 0.0):
        return []
    c = (x3 - x1) * dy - (y3 - y1) * dx
    if abs(c) > dist_tol * math.hypot(dx, dy):
        return []
    # 共线：投影到主轴求重叠区间端点
    if abs(dx) >= abs(dy) and dx != 0.0:
        t3 = (x3 - x1) / dx
        t4 = (x4 - x1) / dx
    elif dy != 0.0:
        t3 = (y3 - y1) / dy
        t4 = (y4 - y1) / dy
    else:
        return []
    lo, hi = (t3, t4) if t3 <= t4 else (t4, t3)
    a = lo if lo > 0.0 else 0.0
    b = hi if hi < 1.0 else 1.0
    if a > b:
        return []
    res = []
    for t in (a, b):
        px = x1 + t * dx
        py = y1 + t * dy
        if abs(ex) >= abs(ey) and ex != 0.0:
            u = (px - x3) / ex
        elif ey != 0.0:
            u = (py - y3) / ey
        else:
            u = 0.0
        res.append((t, u))
    return res


# ---------------------------------------------------------------------------
# 矩形裁剪（Sutherland–Hodgman）
# ---------------------------------------------------------------------------

def _sh_halfplane(poly, axis, value, keep_le):
    """用半平面裁剪凸边界。axis=0 为 x，1 为 y；keep_le=True 保留 coord<=value。"""
    out = []
    n = len(poly)
    if n == 0:
        return out
    for i in range(n):
        x2, y2 = poly[i]
        x1, y1 = poly[i - 1]
        c1 = x1 if axis == 0 else y1
        c2 = x2 if axis == 0 else y2
        if keep_le:
            in1 = c1 <= value
            in2 = c2 <= value
        else:
            in1 = c1 >= value
            in2 = c2 >= value
        if in2:
            if not in1:
                t = (value - c1) / (c2 - c1)
                out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            out.append((x2, y2))
        elif in1:
            t = (value - c1) / (c2 - c1)
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return out


# ---------------------------------------------------------------------------
# 凹多边形对凹多边形（Weiler–Atherton）
# ---------------------------------------------------------------------------

def _weiler_atherton(subj, clip):
    ns = len(subj)
    nc = len(clip)
    scale = 1.0
    for x, y in subj:
        if abs(x) > scale:
            scale = abs(x)
        if abs(y) > scale:
            scale = abs(y)
    for x, y in clip:
        if abs(x) > scale:
            scale = abs(x)
        if abs(y) > scale:
            scale = abs(y)
    dist_tol = 1e-9 * scale

    s_hits = [[] for _ in range(ns)]
    c_hits = [[] for _ in range(nc)]
    inters = []  # iid -> (x, y)
    key_to_id = {}

    def inter_id(x, y):
        k = (round(x / dist_tol), round(y / dist_tol))
        iid = key_to_id.get(k)
        if iid is None:
            iid = len(inters)
            inters.append((x, y))
            key_to_id[k] = iid
        return iid

    # —— 均匀网格粗筛：clip 边入桶，subject 边查询，避免 O(n*m) ——
    cxs = [p[0] for p in clip]
    cys = [p[1] for p in clip]
    minx = min(cxs)
    maxx = max(cxs)
    miny = min(cys)
    maxy = max(cys)
    G = max(32, min(64, int(math.sqrt(nc))))
    cell = max(maxx - minx, maxy - miny) / G
    if cell <= 0.0:
        cell = 1.0
    grid = {}
    cbx0 = [0.0] * nc
    cby0 = [0.0] * nc
    cbx1 = [0.0] * nc
    cby1 = [0.0] * nc
    ccx = [0.0] * nc   # clip 边起点
    ccy = [0.0] * nc
    cex = [0.0] * nc   # clip 边方向
    cey = [0.0] * nc
    for j in range(nc):
        x1, y1 = clip[j]
        x2, y2 = clip[(j + 1) % nc]
        ccx[j] = x1
        ccy[j] = y1
        cex[j] = x2 - x1
        cey[j] = y2 - y1
        if x1 < x2:
            cbx0[j] = x1
            cbx1[j] = x2
        else:
            cbx0[j] = x2
            cbx1[j] = x1
        if y1 < y2:
            cby0[j] = y1
            cby1[j] = y2
        else:
            cby0[j] = y2
            cby1[j] = y1
        gx0 = int((cbx0[j] - minx) / cell)
        gx1 = int((cbx1[j] - minx) / cell)
        gy0 = int((cby0[j] - miny) / cell)
        gy1 = int((cby1[j] - miny) / cell)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                grid.setdefault((gx, gy), []).append(j)

    grid_get = grid.get
    tol = _T_TOL
    stamp = [-1] * nc
    for i in range(ns):
        p2 = subj[i + 1] if i + 1 < ns else subj[0]
        x1, y1 = subj[i]
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        if x1 < x2:
            sx0 = x1
            sx1 = x2
        else:
            sx0 = x2
            sx1 = x1
        if y1 < y2:
            sy0 = y1
            sy1 = y2
        else:
            sy0 = y2
            sy1 = y1
        gx0 = int((sx0 - minx) / cell)
        gx1 = int((sx1 - minx) / cell)
        gy0 = int((sy0 - miny) / cell)
        gy1 = int((sy1 - miny) / cell)
        if gx1 < 0 or gy1 < 0 or gx0 > G or gy0 > G:
            continue
        if gx0 < 0:
            gx0 = 0
        if gy0 < 0:
            gy0 = 0
        if gx1 > G:
            gx1 = G
        if gy1 > G:
            gy1 = G
        s_hit = s_hits[i]
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                for j in grid_get((gx, gy), ()):
                    if stamp[j] == i:
                        continue
                    stamp[j] = i
                    # 内联包围盒预筛
                    if cbx0[j] > sx1 or cbx1[j] < sx0 or \
                            cby0[j] > sy1 or cby1[j] < sy0:
                        continue
                    # 内联穿越测试（常见情形）；平行/共线走 _seg_seg
                    ex = cex[j]
                    ey = cey[j]
                    denom = dx * ey - dy * ex
                    if denom != 0.0:
                        wx = ccx[j] - x1
                        wy = ccy[j] - y1
                        t = (wx * ey - wy * ex) / denom
                        if t < -tol or t > 1.0 + tol:
                            continue
                        u = (wx * dy - wy * dx) / denom
                        if u < -tol or u > 1.0 + tol:
                            continue
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        if u < 0.0:
                            u = 0.0
                        elif u > 1.0:
                            u = 1.0
                        iid = inter_id(x1 + t * dx, y1 + t * dy)
                        s_hit.append((t, iid))
                        c_hits[j].append((u, iid))
                    else:
                        for t, u in _seg_seg((x1, y1), p2,
                                             (ccx[j], ccy[j]),
                                             (ccx[j] + ex, ccy[j] + ey),
                                             dist_tol):
                            iid = inter_id(x1 + t * dx, y1 + t * dy)
                            s_hit.append((t, iid))
                            c_hits[j].append((u, iid))

    # —— 构建双方节点序列（顶点 + 按边参数排序的交点），t≈1 归下一条边 ——
    def build_nodes(poly, hits, n):
        nodes = []  # (x, y, iid 或 -1)
        loc = {}    # iid -> (edge_i, t)
        for i in range(n):
            nodes.append((poly[i][0], poly[i][1], -1))
            prev = -1
            for t, iid in sorted(hits[i]):
                if t >= 1.0 - _T_TOL:
                    continue
                if iid == prev or iid in loc:
                    continue
                prev = iid
                loc[iid] = (i, t)
                x, y = inters[iid]
                nodes.append((x, y, iid))
        return nodes, loc

    s_nodes, s_loc = build_nodes(subj, s_hits, ns)
    c_nodes, c_loc = build_nodes(clip, c_hits, nc)

    s_pos = {}
    for idx, nd in enumerate(s_nodes):
        if nd[2] >= 0:
            s_pos[nd[2]] = idx
    c_pos = {}
    for idx, nd in enumerate(c_nodes):
        if nd[2] >= 0:
            c_pos[nd[2]] = idx

    # —— 交点分类：利用两边方向叉积做局部入/出判定（O(1) 每交点）——
    # clip 逆时针时内部在边的左侧；subject 边方向 d 与 clip 边方向 e 的
    # 叉积 cross(e, d) > 0 表示 subject 由外向内穿越（入点），反之出点；
    # 接近平行（含共线重叠）视为相切，追踪时不切换多边形。
    clip_ccw = _signed_area2(clip) > 0.0
    info = {}
    for iid, (ei, t) in s_loc.items():
        cj = c_loc.get(iid)
        if cj is None:
            info[iid] = 0
            continue
        ej = cj[0]
        sx1, sy1 = subj[ei]
        sx2, sy2 = subj[(ei + 1) % ns]
        cx1, cy1 = clip[ej]
        cx2, cy2 = clip[(ej + 1) % nc]
        dx = sx2 - sx1
        dy = sy2 - sy1
        ex = cx2 - cx1
        ey = cy2 - cy1
        cross = ex * dy - ey * dx
        tol = 1e-12 * math.hypot(dx, dy) * math.hypot(ex, ey)
        if cross > tol:
            info[iid] = 1 if clip_ccw else -1
        elif cross < -tol:
            info[iid] = -1 if clip_ccw else 1
        else:
            info[iid] = 0   # 相切（不切换多边形）

    # —— 从每个未访问入点出发追踪闭环 ——
    results = []
    used = set()
    max_steps = 4 * (len(s_nodes) + len(c_nodes)) + 64

    for idx, nd in enumerate(s_nodes):
        start_iid = nd[2]
        if start_iid < 0 or info.get(start_iid) != 1 or start_iid in used:
            continue
        ring = []
        start = (0, idx)  # (0=subject, 1=clip)
        pos = start
        first = True
        steps = 0
        ok = True
        while ok:
            if not first and pos == start:
                break
            first = False
            nodes = s_nodes if pos[0] == 0 else c_nodes
            x, y, cur = nodes[pos[1]]
            ring.append((x, y))
            if cur >= 0:
                used.add(cur)
            # 沿当前多边形前进到下一个交点
            k = pos[1]
            while True:
                k = (k + 1) % len(nodes)
                x, y, cur = nodes[k]
                steps += 1
                if steps > max_steps:
                    ok = False
                    break
                if cur < 0:
                    ring.append((x, y))
                else:
                    break
            if not ok:
                break
            pos = (pos[0], k)
            if pos == start:
                break
            x, y, cur = nodes[k]
            ring.append((x, y))
            used.add(cur)
            if info.get(cur, 0) == 0:
                continue  # 相切：不切换
            pos = (1 - pos[0], (c_pos if pos[0] == 0 else s_pos)[cur])
        if not ok:
            continue
        cleaned = _dedupe(ring)
        if len(cleaned) >= 3 and _signed_area2(cleaned) != 0.0:
            results.append(cleaned)

    # —— 无有效穿越交点：包含或相离 ——
    if not results and not any(v != 0 for v in info.values()):
        if _point_in_poly(subj[0], clip):
            return [list(subj)]
        if _point_in_poly(clip[0], subj):
            return [list(clip)]
        return []
    return results


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def clip_polygon(subject, clip):
    """Weiler–Atherton 凹多边形相交裁剪。

    返回多边形环列表（每环为 ``(x, y)`` 顶点列表，可能多环或为空）。
    顶点数 < 3 返回 []；零面积多边形或非有限坐标抛出 ValueError。
    """
    subj = _validate_poly(subject)
    clp = _validate_poly(clip)
    if len(subj) < 3 or len(clp) < 3:
        return []
    if _signed_area2(subj) == 0.0 or _signed_area2(clp) == 0.0:
        raise ValueError("多边形退化（面积为零）")
    return _weiler_atherton(subj, clp)


def polygon_area(poly):
    """多边形面积（顶点数 < 3 时为 0）。"""
    pts = _validate_poly(poly)
    if len(pts) < 3:
        return 0.0
    return abs(_signed_area2(pts)) / 2.0


def intersection_area(a, b):
    """两（可凹）多边形交集面积。"""
    return sum(polygon_area(r) for r in clip_polygon(a, b))


def union_area(a, b):
    """两（可凹）多边形并集面积；与交集面积之和严格等于面积之和。"""
    return polygon_area(a) + polygon_area(b) - intersection_area(a, b)


class Clipper:
    """多边形裁剪器。无内部状态，可安全并发使用。"""

    def clip(self, poly, rect):
        """用轴对齐矩形 ``(xmin, ymin, xmax, ymax)`` 裁剪多边形。

        返回裁剪后的顶点列表（可能为空）。顶点数 < 3 返回 []；
        零面积多边形、非法矩形、非有限坐标抛出 ValueError。
        """
        xmin, ymin, xmax, ymax = _validate_rect(rect)
        pts = _validate_poly(poly)
        if len(pts) < 3:
            return []
        if _signed_area2(pts) == 0.0:
            raise ValueError("多边形退化（面积为零）")
        out = _sh_halfplane(pts, 0, xmin, False)
        out = _sh_halfplane(out, 0, xmax, True)
        out = _sh_halfplane(out, 1, ymin, False)
        out = _sh_halfplane(out, 1, ymax, True)
        out = _dedupe(out)
        if len(out) < 3 or _signed_area2(out) == 0.0:
            return []
        return out

    def clip_poly(self, subject, clip):
        """凹多边形对凹多边形相交裁剪（Weiler–Atherton），返回环列表。"""
        return clip_polygon(subject, clip)

    def intersect_area(self, a, b):
        """两多边形交集面积。"""
        return intersection_area(a, b)

    def union_area(self, a, b):
        """两多边形并集面积（与交集面积守恒）。"""
        return union_area(a, b)
