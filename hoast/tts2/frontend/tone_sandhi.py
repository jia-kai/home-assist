# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE-APACHE-2.0.txt.
# Modified for hoast: typed rules, explicit segment containers, no swallowed errors.
"""Paddle/Melo Mandarin tone-sandhi rules and complete neutral-tone lexicon."""

from typing import Any

# Keep the complete upstream lexicon compact and auditable against its source.
# ruff: noqa: SIM905
NEUTRAL_WORDS: frozenset[str] = frozenset(
    """
麻烦 麻利 鸳鸯 高粱 骨头 骆驼 马虎 首饰 馒头 馄饨 风筝 难为 队伍 阔气 闺女 门道
锄头 铺盖 铃铛 铁匠 钥匙 里脊 里头 部分 那么 道士 造化 迷糊 连累 这么 这个 运气
过去 软和 转悠 踏实 跳蚤 跟头 趔趄 财主 豆腐 讲究 记性 记号 认识 规矩 见识 裁缝
补丁 衣裳 衣服 衙门 街坊 行李 行当 蛤蟆 蘑菇 薄荷 葫芦 葡萄 萝卜 荸荠 苗条 苗头
苍蝇 芝麻 舒服 舒坦 舌头 自在 膏药 脾气 脑袋 脊梁 能耐 胳膊 胭脂 胡萝 胡琴 胡同
聪明 耽误 耽搁 耷拉 耳朵 老爷 老实 老婆 老头 老太 翻腾 罗嗦 罐头 编辑 结实 红火
累赘 糨糊 糊涂 精神 粮食 簸箕 篱笆 算计 算盘 答应 笤帚 笑语 笑话 窟窿 窝囊 窗户
稳当 稀罕 称呼 秧歌 秀气 秀才 福气 祖宗 砚台 码头 石榴 石头 石匠 知识 眼睛 眯缝
眨巴 眉毛 相声 盘算 白净 痢疾 痛快 疟疾 疙瘩 疏忽 畜生 生意 甘蔗 琵琶 琢磨 琉璃
玻璃 玫瑰 玄乎 狐狸 状元 特务 牲口 牙碜 牌楼 爽快 爱人 热闹 烧饼 烟筒 烂糊 点心
炊帚 灯笼 火候 漂亮 滑溜 溜达 温和 清楚 消息 浪头 活泼 比方 正经 欺负 模糊 槟榔
棺材 棒槌 棉花 核桃 栅栏 柴火 架势 枕头 枇杷 机灵 本事 木头 木匠 朋友 月饼 月亮
暖和 明白 时候 新鲜 故事 收拾 收成 提防 挖苦 挑剔 指甲 指头 拾掇 拳头 拨弄 招牌
招呼 抬举 护士 折腾 扫帚 打量 打算 打点 打扮 打听 打发 扎实 扁担 戒指 懒得 意识
意思 情形 悟性 怪物 思量 怎么 念头 念叨 快活 忙活 志气 心思 得罪 张罗 弟兄 开通
应酬 庄稼 干事 帮手 帐篷 希罕 师父 师傅 巴结 巴掌 差事 工夫 岁数 屁股 尾巴 少爷
小气 小伙 将就 对头 对付 寡妇 家伙 客气 实在 官司 学问 学生 字号 嫁妆 媳妇 媒人
婆家 娘家 委屈 姑娘 姐夫 妯娌 妥当 妖精 奴才 女婿 头发 太阳 大爷 大方 大意 大夫
多少 多么 外甥 壮实 地道 地方 在乎 困难 嘴巴 嘱咐 嘟囔 嘀咕 喜欢 喇嘛 喇叭 商量
唾沫 哑巴 哈欠 哆嗦 咳嗽 和尚 告诉 告示 含糊 吓唬 后头 名字 名堂 合同 吆喝 叫唤
口袋 厚道 厉害 千斤 包袱 包涵 匀称 勤快 动静 动弹 功夫 力气 前头 刺猬 刺激 别扭
利落 利索 利害 分析 出息 凑合 凉快 冷战 冤枉 冒失 养活 关系 先生 兄弟 便宜 使唤
佩服 作坊 体面 位置 似的 伙计 休息 什么 人家 亲戚 亲家 交情 云彩 事情 买卖 主意
丫头 丧气 两口 东西 东家 世故 不由 不在 下水 下巴 上头 上司 丈夫 丈人 一辈 那个
菩萨 父亲 母亲 咕噜 邋遢 费用 冤家 甜头 介绍 荒唐 大人 泥鳅 幸福 熟悉 计划 扑腾
蜡烛 姥爷 照顾 喉咙 吉他 弄堂 蚂蚱 凤凰 拖沓 寒碜 糟蹋 倒腾 报复 逻辑 盘缠 喽啰
牢骚 咖喱 扫把 惦记
""".split()
)
NON_NEUTRAL_WORDS: frozenset[str] = frozenset(
    [
        "男子",
        "女子",
        "分子",
        "原子",
        "量子",
        "莲子",
        "石子",
        "瓜子",
        "电子",
        "人人",
        "虎虎",
    ]
)


class ToneSandhi:
    """Apply upstream lexical, reduplication, 一/不 and third-tone rules."""

    jieba: Any
    """Jieba module supplying upstream search segmentation."""

    lazy_pinyin: Any
    """Pypinyin callable supplying raw finals for merge decisions."""

    style: Any
    """Pypinyin Style enum."""

    def __init__(self) -> None:
        """Load Chinese-only tooling lazily; no external model resources are read."""
        # Chinese dependencies must not be imported for English-only inference.
        import jieba
        from pypinyin import Style, lazy_pinyin

        self.jieba = jieba
        self.lazy_pinyin = lazy_pinyin
        self.style = Style

    def _split_word(self, word: str) -> list[str]:
        """Split using the shortest Jieba search subword as in Paddle.

        Args:
            word:
                Nonempty segmented Chinese word.

        """
        first = min(self.jieba.cut_for_search(word), key=len)
        if word.find(first) == 0:
            return [first, word[len(first) :]]
        return [word[: -len(first)], first]

    def _neutral(self, word: str, pos: str, finals: list[str]) -> list[str]:
        """Apply the upstream neutral-tone lexicon and morphological rules.

        Args:
            word:
                Segmented word matching the finals sequence.

            pos:
                Jieba part-of-speech tag.

            finals:
                Mutable per-character finals with numeric tones.

        """
        for index, char in enumerate(word):
            if (
                index > 0
                and char == word[index - 1]
                and pos[0] in {"n", "v", "a"}
                and word not in NON_NEUTRAL_WORDS
            ):
                finals[index] = finals[index][:-1] + "5"
        ge_index = word.find("个")
        if (
            word[-1] in "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶的地得"
            or (
                len(word) > 1
                and word[-1] in "们子"
                and pos in {"r", "n"}
                and word not in NON_NEUTRAL_WORDS
            )
            or (len(word) > 1 and word[-1] in "上下里" and pos in {"s", "l", "f"})
            or (len(word) > 1 and word[-1] in "来去" and word[-2] in "上下进出回过起开")
        ):
            finals[-1] = finals[-1][:-1] + "5"
        elif (
            ge_index >= 1
            and (
                word[ge_index - 1].isnumeric()
                or word[ge_index - 1] in "几有两半多各整每做是"
            )
        ) or word == "个":
            finals[ge_index] = finals[ge_index][:-1] + "5"
        elif word in NEUTRAL_WORDS or word[-2:] in NEUTRAL_WORDS:
            finals[-1] = finals[-1][:-1] + "5"
        parts = self._split_word(word)
        offset = 0
        for part in parts:
            if part in NEUTRAL_WORDS or part[-2:] in NEUTRAL_WORDS:
                finals[offset + len(part) - 1] = (
                    finals[offset + len(part) - 1][:-1] + "5"
                )
            offset += len(part)
        return finals

    def modified_tone(self, word: str, pos: str, finals: list[str]) -> list[str]:
        """Modify a word's finals in upstream 不/一/neutral/third-tone order.

        Args:
            word:
                Nonempty Chinese word or supported punctuation.

            pos:
                Jieba part-of-speech tag.

            finals:
                Per-character finals, changed in place where practical.

        """
        if not word or len(finals) != len(word):
            raise ValueError(f"Invalid sandhi word/finals alignment: {word!r}")
        if len(word) == 3 and word[1] == "不":
            finals[1] = finals[1][:-1] + "5"
        else:
            for index, char in enumerate(word[:-1]):
                if char == "不" and finals[index + 1][-1] == "4":
                    finals[index] = finals[index][:-1] + "2"
        if "一" in word and all(char.isnumeric() for char in word if char != "一"):
            pass
        elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            finals[1] = finals[1][:-1] + "5"
        elif word.startswith("第一"):
            finals[1] = finals[1][:-1] + "1"
        else:
            for index, char in enumerate(word[:-1]):
                if char == "一":
                    if finals[index + 1][-1] == "4":
                        finals[index] = finals[index][:-1] + "2"
                    elif word[index + 1] not in "：，；。？！“”‘’':,;.?!":
                        finals[index] = finals[index][:-1] + "4"
        finals = self._neutral(word, pos, finals)
        if len(word) == 2 and self._all_three(finals):
            finals[0] = finals[0][:-1] + "2"
        elif len(word) == 3:
            parts = self._split_word(word)
            boundary = len(parts[0])
            if self._all_three(finals):
                if boundary == 2:
                    finals[0] = finals[0][:-1] + "2"
                    finals[1] = finals[1][:-1] + "2"
                elif boundary == 1:
                    finals[1] = finals[1][:-1] + "2"
            else:
                pieces = [finals[:boundary], finals[boundary:]]
                for index, piece in enumerate(pieces):
                    if self._all_three(piece) and len(piece) == 2:
                        piece[0] = piece[0][:-1] + "2"
                    elif (
                        index == 1
                        and not self._all_three(piece)
                        and piece[0][-1] == "3"
                        and pieces[0][-1][-1] == "3"
                    ):
                        pieces[0][-1] = pieces[0][-1][:-1] + "2"
                finals = [final for piece in pieces for final in piece]
        elif len(word) == 4:
            pieces = [finals[:2], finals[2:]]
            for piece in pieces:
                if self._all_three(piece):
                    piece[0] = piece[0][:-1] + "2"
            finals = [final for piece in pieces for final in piece]
        return finals

    @staticmethod
    def _all_three(finals: list[str]) -> bool:
        """Return whether all final tones are three, including an empty list.

        Args:
            finals:
                Pinyin finals with trailing tone digits.

        """
        return all(final[-1] == "3" for final in finals)

    @staticmethod
    def _reduplicated(word: str) -> bool:
        """Identify a two-character reduplication excluded from tone merging.

        Args:
            word:
                Segmented word spelling.

        """
        return len(word) == 2 and word[0] == word[1]

    def _merge_three(
        self, segments: list[tuple[str, str]], boundary_only: bool
    ) -> list[tuple[str, str]]:
        """Merge short adjacent third-tone words under one of Paddle's two rules.

        Args:
            segments:
                Ordered word/POS pairs.

            boundary_only:
                Test boundary tones rather than all tones in both words.

        """
        finals = [
            self.lazy_pinyin(
                word, neutral_tone_with_five=True, style=self.style.FINALS_TONE3
            )
            for word, _ in segments
        ]
        result: list[tuple[str, str]] = []
        merged = [False] * len(segments)
        for index, (word, pos) in enumerate(segments):
            match = False
            if index > 0 and not merged[index - 1]:
                if boundary_only:
                    match = (
                        finals[index - 1][-1][-1] == "3" and finals[index][0][-1] == "3"
                    )
                else:
                    match = self._all_three(finals[index - 1]) and self._all_three(
                        finals[index]
                    )
            if (
                match
                and not self._reduplicated(segments[index - 1][0])
                and len(segments[index - 1][0]) + len(word) <= 3
            ):
                result[-1] = (result[-1][0] + word, result[-1][1])
                merged[index] = True
            else:
                result.append((word, pos))
        return result

    def pre_merge_for_modify(
        self, segments: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Merge 不, 一, reduplications, third-tone runs and 儿 in upstream order.

        Args:
            segments:
                Jieba word/POS pairs, copied rather than mutated.

        """
        result: list[tuple[str, str]] = []
        pending = ""
        for word, pos in segments:
            if pending == "不":
                word = pending + word
            if word != "不":
                result.append((word, pos))
            pending = word
        if pending == "不":
            result.append((pending, "d"))
        segments = result
        result = []
        for index, (word, pos) in enumerate(segments):
            if (
                index > 0
                and word == "一"
                and index + 1 < len(segments)
                and segments[index - 1][0] == segments[index + 1][0]
                and segments[index - 1][1] == "v"
            ):
                result[-1] = (result[-1][0] + "一" + result[-1][0], result[-1][1])
            elif (
                index >= 2
                and segments[index - 1][0] == "一"
                and segments[index - 2][0] == word
                and pos == "v"
            ):
                continue
            else:
                result.append((word, pos))
        segments = result
        result = []
        for word, pos in segments:
            if result and result[-1][0] == "一":
                result[-1] = (result[-1][0] + word, result[-1][1])
            else:
                result.append((word, pos))
        segments = result
        result = []
        for word, pos in segments:
            if result and word == result[-1][0]:
                result[-1] = (result[-1][0] + word, result[-1][1])
            else:
                result.append((word, pos))
        segments = self._merge_three(result, boundary_only=False)
        segments = self._merge_three(segments, boundary_only=True)
        result = []
        for index, (word, pos) in enumerate(segments):
            if index > 0 and word == "儿" and segments[index - 1][0] != "#":
                result[-1] = (result[-1][0] + word, result[-1][1])
            else:
                result.append((word, pos))
        return result
