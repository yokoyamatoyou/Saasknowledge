# RAG検索精度向上 - Codex対応実装ガイド

## 実装概要
既存の`HybridSearchEngine`クラスを拡張し、バージョン管理、ルール標準化、階層的検索を実装します。

## 1. 拡張メタデータスキーマ（metadata/*.json）

```python
# metadata/chunk_001.json の新しい構造
{
    # 既存フィールド
    "id": "chunk_001",
    "filename": "就業規則_営業部_2024",
    "token_count": 850,
    "char_count": 2100,
    "created_at": "2024-01-15 10:30:00",
    
    # 新規追加フィールド - バージョン管理
    "version_info": {
        "version": "2.1.0",                    # セマンティックバージョニング
        "effective_date": "2024-01-01",        # 施行日
        "expiry_date": null,                   # 有効期限（nullは無期限）
        "supersedes": ["chunk_old_001"],       # 置き換える旧バージョンのID
        "superseded_by": null,                 # 新バージョンのID（最新版はnull）
        "status": "active"                     # active|deprecated|draft
    },
    
    # 新規追加フィールド - 階層情報
    "hierarchy_info": {
        "approval_level": "department",        # company|department|local
        "department": "営業部",                 # 部門名
        "location": "東京本社",                 # 営業所・拠点
        "authority_score": 0.7                 # 権威度スコア（0.0-1.0）
    },
    
    # 新規追加フィールド - ルール情報
    "rule_info": {
        "contains_rules": true,
        "rule_types": ["見積り条件", "承認権限"],
        "extracted_rules": [
            {
                "rule_id": "r001",
                "rule_type": "見積り条件",
                "rule_content": "100万円以下の見積りは課長承認",
                "conditions": ["金額 <= 1000000", "部門 = 営業"],
                "values": {"threshold_amount": 1000000, "approver": "課長"},
                "confidence": 0.95
            }
        ]
    },
    
    # 既存のmeta_info（generate_chunk_metadataで生成）
    "meta_info": {
        "summary": "営業部の見積り承認権限に関する規定",
        "keywords": ["見積り", "承認", "営業部", "権限"],
        "tags": ["営業規定", "承認フロー"],
        "search_queries": ["見積り承認", "営業部の権限"],
        "synonyms": {"見積り": ["見積", "見積もり", "見積書"]},
        "semantic_connections": ["決裁権限", "稟議", "承認フロー"],
        "mini_context": "営業部における見積り金額別の承認権限を定めた規定"
    }
}
```

## 2. 拡張HybridSearchEngineクラス

```python
# search_engine.py への追加実装

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

class EnhancedHybridSearchEngine(HybridSearchEngine):
    """
    バージョン管理、ルール抽出、階層的検索を追加した拡張検索エンジン
    
    主な拡張機能:
    1. タイムスタンプベースの重み付け
    2. バージョン管理とフィルタリング  
    3. ルール抽出と矛盾検出
    4. 階層的検索（全社→部門→営業所）
    5. 意図認識による動的パラメータ調整
    """
    
    def __init__(self, kb_path: str):
        # 親クラスの初期化
        super().__init__(kb_path)
        
        # 拡張機能用の追加属性
        self.version_graph = {}  # chunk_id -> version_info のマッピング
        self.rule_index = {}     # rule_type -> [chunk_ids] のマッピング
        self.hierarchy_index = {} # approval_level -> [chunk_ids] のマッピング
        
        # 拡張インデックスの構築
        self._build_extended_indexes()
        
        logger.info(f"EnhancedHybridSearchEngine初期化完了: "
                   f"バージョン管理対象: {len(self.version_graph)}件, "
                   f"ルール種別: {len(self.rule_index)}種類")
    
    def _build_extended_indexes(self):
        """拡張メタデータからインデックスを構築"""
        for chunk in self.chunks:
            chunk_id = chunk["id"]
            metadata = chunk.get("metadata", {})
            
            # バージョン情報の収集
            if "version_info" in metadata:
                self.version_graph[chunk_id] = metadata["version_info"]
            
            # ルール情報のインデックス化
            if metadata.get("rule_info", {}).get("contains_rules"):
                for rule_type in metadata["rule_info"].get("rule_types", []):
                    if rule_type not in self.rule_index:
                        self.rule_index[rule_type] = []
                    self.rule_index[rule_type].append(chunk_id)
            
            # 階層情報のインデックス化
            if "hierarchy_info" in metadata:
                approval_level = metadata["hierarchy_info"].get("approval_level")
                if approval_level:
                    if approval_level not in self.hierarchy_index:
                        self.hierarchy_index[approval_level] = []
                    self.hierarchy_index[approval_level].append(chunk_id)
    
    def calculate_recency_weight(self, chunk_metadata: dict, 
                                query_date: Optional[datetime] = None) -> float:
        """
        文書の新しさに基づいて重みを計算
        
        Args:
            chunk_metadata: チャンクのメタデータ
            query_date: 基準日（省略時は現在日時）
            
        Returns:
            float: 0.0-1.0の範囲の新しさスコア
        """
        if query_date is None:
            query_date = datetime.now()
        
        # effective_dateを取得
        version_info = chunk_metadata.get("version_info", {})
        effective_date_str = version_info.get("effective_date")
        
        if not effective_date_str:
            # created_atをフォールバック
            created_at_str = chunk_metadata.get("created_at")
            if created_at_str:
                effective_date_str = created_at_str.split()[0]  # 日付部分のみ
            else:
                return 0.5  # デフォルト値
        
        try:
            # 日付文字列をdatetimeオブジェクトに変換
            if isinstance(effective_date_str, str):
                effective_date = datetime.strptime(effective_date_str, "%Y-%m-%d")
            else:
                effective_date = effective_date_str
                
            # 経過日数を計算
            days_old = (query_date - effective_date).days
            
            # 指数関数的減衰
            # 30日で約0.97、365日で約0.69、730日で約0.48になる減衰率
            decay_rate = 0.001
            recency_weight = np.exp(-decay_rate * days_old)
            
            # 有効期限チェック
            expiry_date_str = version_info.get("expiry_date")
            if expiry_date_str:
                expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d")
                if query_date > expiry_date:
                    recency_weight *= 0.1  # 期限切れは大幅に減点
            
            return float(recency_weight)
            
        except Exception as e:
            logger.warning(f"日付解析エラー (chunk_id: {chunk_metadata.get('id')}): {e}")
            return 0.5
    
    def filter_latest_versions(self, chunks: List[dict]) -> List[dict]:
        """
        チャンクリストから最新バージョンのみをフィルタリング
        
        Args:
            chunks: フィルタリング対象のチャンクリスト
            
        Returns:
            List[dict]: 最新バージョンのチャンクのみのリスト
        """
        # superseded_byがnullのもの（= 最新版）を優先
        latest_chunks = []
        deprecated_ids = set()
        
        # まず、置き換えられたチャンクIDを収集
        for chunk in chunks:
            version_info = chunk.get("metadata", {}).get("version_info", {})
            superseded_by = version_info.get("superseded_by")
            if superseded_by:
                deprecated_ids.add(chunk["id"])
        
        # 最新版のみを選択
        for chunk in chunks:
            chunk_id = chunk["id"]
            version_info = chunk.get("metadata", {}).get("version_info", {})
            status = version_info.get("status", "active")
            
            # deprecatedまたは置き換えられたものは除外
            if status == "deprecated" or chunk_id in deprecated_ids:
                continue
                
            latest_chunks.append(chunk)
        
        return latest_chunks
    
    def detect_rule_conflicts(self, chunks: List[dict], client=None) -> List[dict]:
        """
        チャンク間のルール矛盾を検出
        
        Args:
            chunks: 検査対象のチャンクリスト
            client: OpenAIクライアント
            
        Returns:
            List[dict]: 検出された矛盾のリスト
        """
        if client is None:
            client = get_openai_client()
            if client is None:
                logger.warning("ルール矛盾検出: OpenAIクライアントが利用できません")
                return []
        
        conflicts = []
        rules_by_type = {}
        
        # ルールをタイプ別に収集
        for chunk in chunks:
            rule_info = chunk.get("metadata", {}).get("rule_info", {})
            if not rule_info.get("contains_rules"):
                continue
                
            for rule in rule_info.get("extracted_rules", []):
                rule_type = rule.get("rule_type")
                if rule_type:
                    if rule_type not in rules_by_type:
                        rules_by_type[rule_type] = []
                    rules_by_type[rule_type].append({
                        "chunk_id": chunk["id"],
                        "rule": rule,
                        "source": chunk.get("metadata", {}).get("hierarchy_info", {})
                    })
        
        # 同じタイプのルール間で矛盾をチェック
        for rule_type, rule_list in rules_by_type.items():
            if len(rule_list) < 2:
                continue
                
            # GPT-4を使用して矛盾を検出
            rules_text = json.dumps(rule_list, ensure_ascii=False, indent=2)
            
            prompt = f"""
            以下の{rule_type}に関するルールを分析し、矛盾や不整合を検出してください。
            
            ルール一覧:
            {rules_text}
            
            以下のJSON形式で矛盾を報告してください:
            {{
                "has_conflict": true/false,
                "conflicts": [
                    {{
                        "rule_type": "{rule_type}",
                        "conflicting_chunks": ["chunk_id1", "chunk_id2"],
                        "explanation": "矛盾の説明",
                        "recommendation": "推奨される解決方法"
                    }}
                ]
            }}
            """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "ビジネスルールの矛盾検出専門家"},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                result = json.loads(response.choices[0].message.content)
                if result.get("has_conflict"):
                    conflicts.extend(result.get("conflicts", []))
                    
            except Exception as e:
                logger.error(f"ルール矛盾検出エラー (rule_type: {rule_type}): {e}")
        
        return conflicts
    
    def classify_query_intent(self, query: str, client=None) -> dict:
        """
        検索クエリの意図を分析
        
        Args:
            query: 検索クエリ
            client: OpenAIクライアント
            
        Returns:
            dict: 意図分析結果
        """
        if client is None:
            client = get_openai_client()
            if client is None:
                # デフォルト値を返す
                return {
                    "primary_intent": "general",
                    "temporal_requirement": "any",
                    "scope": "company_wide",
                    "needs_latest": False
                }
        
        prompt = f"""
        以下の検索クエリの意図を分析してください。
        クエリ: "{query}"
        
        JSON形式で以下の情報を返してください:
        {{
            "primary_intent": "latest_info|procedure|comparison|definition|general",
            "temporal_requirement": "latest|historical|any",
            "scope": "company_wide|department|specific_location",
            "needs_latest": true/false,
            "rule_type": "見積り条件|承認権限|手続き|その他|null",
            "keywords": ["抽出されたキーワード"]
        }}
        
        判断基準:
        - "最新の"、"現在の"、"今の" → temporal_requirement: "latest"
        - "全社"、"会社全体" → scope: "company_wide"
        - "営業部の"、"東京支店の" → scope: "department" or "specific_location"
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "検索意図分析の専門家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # 一貫性のため低めに設定
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"クエリ意図分析エラー: {e}")
            return {
                "primary_intent": "general",
                "temporal_requirement": "any", 
                "scope": "company_wide",
                "needs_latest": False
            }
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.15,
               vector_weight: float = None, bm25_weight: float = None,
               client=None) -> Tuple[List[dict], bool]:
        """
        拡張検索メソッド - バージョン管理と階層を考慮
        
        処理フロー:
        1. クエリ意図を分析
        2. 基本検索（親クラスのロジック）を実行
        3. バージョンフィルタリング
        4. 階層的重み付け
        5. ルール矛盾チェック
        6. 最終スコアリングと並び替え
        """
        logger.info(f"拡張検索開始: query='{query}'")
        
        # Step 1: クエリ意図分析
        intent = self.classify_query_intent(query, client)
        logger.info(f"クエリ意図: {intent}")
        
        # Step 2: 動的な重み調整
        if vector_weight is None or bm25_weight is None:
            vector_weight, bm25_weight = compute_hybrid_weights(len(self.chunks))
        
        # 意図に基づいて重みを調整
        recency_weight = 0.0
        hierarchy_weight = 0.0
        
        if intent.get("needs_latest") or intent.get("temporal_requirement") == "latest":
            recency_weight = 0.3
            vector_weight *= 0.7
            bm25_weight *= 0.7
            
        if intent.get("scope") == "company_wide":
            hierarchy_weight = 0.2
            vector_weight *= 0.8
            bm25_weight *= 0.8
        
        # Step 3: 基本検索の実行（親クラスのメソッドを直接呼び出し）
        base_results, not_found = super().search(
            query, top_k=top_k*3,  # 後でフィルタリングするため多めに取得
            threshold=threshold/2,  # 閾値も緩めに
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            client=client
        )
        
        if not_found or not base_results:
            logger.info("基本検索で結果が見つかりませんでした")
            return [], True
        
        # Step 4: 拡張スコアリング
        enhanced_results = []
        
        for result in base_results:
            chunk_id = result["id"]
            chunk_metadata = result.get("metadata", {})
            
            # 新しさスコア
            recency_score = 0.0
            if recency_weight > 0:
                recency_score = self.calculate_recency_weight(chunk_metadata)
            
            # 階層スコア
            hierarchy_score = 0.0
            if hierarchy_weight > 0:
                hierarchy_info = chunk_metadata.get("hierarchy_info", {})
                approval_level = hierarchy_info.get("approval_level", "local")
                
                # 階層レベルに応じたスコア
                hierarchy_scores_map = {
                    "company": 1.0,     # 全社規定が最優先
                    "department": 0.7,  # 部門規定
                    "local": 0.4        # 営業所独自
                }
                hierarchy_score = hierarchy_scores_map.get(approval_level, 0.4)
                
                # 権威度スコアも考慮
                authority = hierarchy_info.get("authority_score", 0.5)
                hierarchy_score = hierarchy_score * 0.7 + authority * 0.3
            
            # 最終スコアの計算
            base_score = result["similarity"]
            final_score = (
                base_score * (1 - recency_weight - hierarchy_weight) +
                recency_score * recency_weight +
                hierarchy_score * hierarchy_weight
            )
            
            # デバッグ情報を追加
            result["score_breakdown"] = {
                "base_score": base_score,
                "recency_score": recency_score,
                "hierarchy_score": hierarchy_score,
                "final_score": final_score,
                "weights": {
                    "base": 1 - recency_weight - hierarchy_weight,
                    "recency": recency_weight,
                    "hierarchy": hierarchy_weight
                }
            }
            
            result["similarity"] = final_score
            enhanced_results.append(result)
        
        # Step 5: バージョンフィルタリング（最新情報が必要な場合）
        if intent.get("needs_latest"):
            # チャンクデータを含む形式に変換
            chunks_with_data = []
            for result in enhanced_results:
                chunk_data = {
                    "id": result["id"],
                    "text": result["text"],
                    "metadata": result["metadata"]
                }
                chunks_with_data.append(chunk_data)
            
            # 最新バージョンのみをフィルタリング
            latest_chunks = self.filter_latest_versions(chunks_with_data)
            latest_ids = {chunk["id"] for chunk in latest_chunks}
            
            # 結果をフィルタリング
            enhanced_results = [r for r in enhanced_results if r["id"] in latest_ids]
        
        # Step 6: ソートと上位K件の選択
        enhanced_results.sort(key=lambda x: x["similarity"], reverse=True)
        final_results = [r for r in enhanced_results if r["similarity"] >= threshold][:top_k]
        
        # Step 7: ルール矛盾チェック（複数の結果がある場合）
        if len(final_results) > 1 and intent.get("rule_type"):
            # 結果をチャンク形式に変換
            result_chunks = []
            for result in final_results:
                chunk_data = {
                    "id": result["id"],
                    "text": result["text"],
                    "metadata": result["metadata"]
                }
                result_chunks.append(chunk_data)
            
            # 矛盾検出
            conflicts = self.detect_rule_conflicts(result_chunks, client)
            
            # 矛盾情報を結果に追加
            if conflicts:
                for result in final_results:
                    result["conflicts"] = [
                        c for c in conflicts 
                        if result["id"] in c.get("conflicting_chunks", [])
                    ]
        
        # ログ出力
        logger.info(f"拡張検索完了: {len(final_results)}件の結果")
        for i, result in enumerate(final_results[:3]):
            logger.info(f"  {i+1}. ID: {result['id']}, "
                       f"最終スコア: {result['similarity']:.4f}, "
                       f"内訳: {result.get('score_breakdown', {})}")
        
        return final_results, len(final_results) == 0


# 使用例とテストコード
def test_enhanced_search():
    """拡張検索エンジンのテストコード"""
    
    # テスト用のクエリ例
    test_queries = [
        {
            "query": "最新の見積り承認権限について",
            "description": "最新情報と特定ルールタイプを要求"
        },
        {
            "query": "全社の就業規則",
            "description": "階層（全社）を指定"
        },
        {
            "query": "営業部の見積もりルール",
            "description": "部門とルールタイプを指定"
        }
    ]
    
    # 検索エンジンの初期化
    kb_path = "path/to/knowledge_base"
    engine = EnhancedHybridSearchEngine(kb_path)
    
    # 各クエリでテスト
    for test_case in test_queries:
        print(f"\n{'='*60}")
        print(f"テストクエリ: {test_case['query']}")
        print(f"説明: {test_case['description']}")
        print(f"{'='*60}")
        
        results, not_found = engine.search(
            test_case['query'],
            top_k=5,
            threshold=0.15
        )
        
        if not_found:
            print("結果が見つかりませんでした")
        else:
            print(f"\n{len(results)}件の結果:")
            for i, result in enumerate(results):
                print(f"\n--- 結果 {i+1} ---")
                print(f"ID: {result['id']}")
                print(f"スコア: {result['similarity']:.4f}")
                
                # スコア内訳
                breakdown = result.get('score_breakdown', {})
                if breakdown:
                    print(f"  - ベーススコア: {breakdown.get('base_score', 0):.4f}")
                    print(f"  - 新しさスコア: {breakdown.get('recency_score', 0):.4f}")
                    print(f"  - 階層スコア: {breakdown.get('hierarchy_score', 0):.4f}")
                
                # バージョン情報
                version_info = result.get('metadata', {}).get('version_info', {})
                if version_info:
                    print(f"バージョン: {version_info.get('version', 'N/A')}")
                    print(f"施行日: {version_info.get('effective_date', 'N/A')}")
                    print(f"ステータス: {version_info.get('status', 'N/A')}")
                
                # 階層情報
                hierarchy_info = result.get('metadata', {}).get('hierarchy_info', {})
                if hierarchy_info:
                    print(f"承認レベル: {hierarchy_info.get('approval_level', 'N/A')}")
                    print(f"部門: {hierarchy_info.get('department', 'N/A')}")
                
                # ルール矛盾
                conflicts = result.get('conflicts', [])
                if conflicts:
                    print(f"⚠️ 検出された矛盾:")
                    for conflict in conflicts:
                        print(f"  - {conflict.get('explanation', 'N/A')}")
                        print(f"    推奨: {conflict.get('recommendation', 'N/A')}")

```

## 3. データ移行スクリプト

既存のナレッジベースに拡張メタデータを追加するためのスクリプト：

```python
# migrate_metadata.py
"""
既存のメタデータファイルに拡張フィールドを追加する移行スクリプト
使用方法: python migrate_metadata.py --kb-path /path/to/knowledge_base
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

logger = logging.getLogger(__name__)

def migrate_metadata_file(metadata_path: Path, chunk_text: str = None):
    """
    単一のメタデータファイルを拡張形式に移行
    
    Args:
        metadata_path: メタデータファイルのパス
        chunk_text: チャンクのテキスト内容（ルール抽出用）
    """
    try:
        # 既存のメタデータを読み込み
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # バージョン情報の追加（デフォルト値）
        if 'version_info' not in metadata:
            metadata['version_info'] = {
                'version': '1.0.0',
                'effective_date': metadata.get('created_at', '2024-01-01').split()[0],
                'expiry_date': None,
                'supersedes': [],
                'superseded_by': None,
                'status': 'active'
            }
        
        # 階層情報の推定
        if 'hierarchy_info' not in metadata:
            # ファイル名やキーワードから推定
            filename = metadata.get('filename', '').lower()
            keywords = metadata.get('meta_info', {}).get('keywords', [])
            
            # デフォルト値
            approval_level = 'local'
            department = '未分類'
            authority_score = 0.5
            
            # キーワードベースの推定
            if any(k in filename for k in ['全社', '会社', 'company']):
                approval_level = 'company'
                authority_score = 0.9
            elif any(k in filename for k in ['部門', '営業部', '総務部', 'department']):
                approval_level = 'department'
                authority_score = 0.7
                
                # 部門名の抽出
                if '営業' in filename:
                    department = '営業部'
                elif '総務' in filename:
                    department = '総務部'
                elif '経理' in filename:
                    department = '経理部'
            
            metadata['hierarchy_info'] = {
                'approval_level': approval_level,
                'department': department,
                'location': '未設定',
                'authority_score': authority_score
            }
        
        # ルール情報の判定（簡易版）
        if 'rule_info' not in metadata:
            rule_keywords = ['承認', '権限', '規定', '手続き', 'ルール', '条件']
            meta_info = metadata.get('meta_info', {})
            
            # キーワードやサマリーからルール含有を判定
            contains_rules = False
            rule_types = []
            
            summary = meta_info.get('summary', '')
            keywords = meta_info.get('keywords', [])
            
            if any(kw in summary or kw in keywords for kw in rule_keywords):
                contains_rules = True
                
                # ルールタイプの推定
                if '見積' in summary or '見積' in keywords:
                    rule_types.append('見積り条件')
                if '承認' in summary or '承認' in keywords:
                    rule_types.append('承認権限')
                if '手続' in summary or '手続' in keywords:
                    rule_types.append('手続き')
            
            metadata['rule_info'] = {
                'contains_rules': contains_rules,
                'rule_types': rule_types,
                'extracted_rules': []  # 後でGPTで抽出
            }
        
        # 更新したメタデータを保存
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"メタデータ移行完了: {metadata_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"メタデータ移行エラー ({metadata_path}): {e}")
        return False

def migrate_knowledge_base(kb_path: str):
    """
    ナレッジベース全体のメタデータを移行
    
    Args:
        kb_path: ナレッジベースのパス
    """
    kb_path = Path(kb_path)
    metadata_path = kb_path / "metadata"
    chunks_path = kb_path / "chunks"
    
    if not metadata_path.exists():
        logger.error(f"メタデータディレクトリが見つかりません: {metadata_path}")
        return
    
    success_count = 0
    error_count = 0
    
    # 各メタデータファイルを処理
    for meta_file in metadata_path.glob("*.json"):
        chunk_id = meta_file.stem
        
        # 対応するチャンクテキストを読み込み（オプション）
        chunk_text = None
        chunk_file = chunks_path / f"{chunk_id}.txt"
        if chunk_file.exists():
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
            except Exception as e:
                logger.warning(f"チャンクテキスト読み込みエラー ({chunk_id}): {e}")
        
        # メタデータを移行
        if migrate_metadata_file(meta_file, chunk_text):
            success_count += 1
        else:
            error_count += 1
    
    logger.info(f"移行完了: 成功 {success_count}件, エラー {error_count}件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="メタデータ移行スクリプト")
    parser.add_argument("--kb-path", required=True, help="ナレッジベースのパス")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    migrate_knowledge_base(args.kb_path)
```

## 4. 実装手順とベストプラクティス

### Step 1: メタデータスキーマの拡張（1週目）
1. `migrate_metadata.py`を実行して既存メタデータを拡張
2. `semantic_chunking`関数を修正してversion_info、hierarchy_infoを追加
3. テストデータで動作確認

### Step 2: 検索エンジンの拡張（2-3週目）
1. `EnhancedHybridSearchEngine`クラスを実装
2. 既存の`HybridSearchEngine`を継承して段階的に機能追加
3. 各メソッドの単体テストを作成

### Step 3: UIの更新（4週目）
1. `app.py`の検索部分を`EnhancedHybridSearchEngine`に切り替え
2. 検索結果表示にバージョン情報、階層情報、矛盾警告を追加
3. 検索オプション（最新のみ、全社規定のみ等）をUIに追加

### Step 4: ルール抽出の高度化（5週目以降）
1. GPT-4を使用した高度なルール抽出を実装
2. バッチ処理で既存チャンクからルールを抽出
3. ルール矛盾レポート機能を追加

## 5. パフォーマンス最適化のヒント

```python
# キャッシュの活用例
from functools import lru_cache
import hashlib

class CachedEnhancedSearchEngine(EnhancedHybridSearchEngine):
    
    @lru_cache(maxsize=1000)
    def _cached_intent_analysis(self, query_hash: str) -> dict:
        """クエリ意図分析結果をキャッシュ"""
        # ハッシュから元のクエリは復元できないため、
        # 実際の実装では別途クエリ管理が必要
        pass
    
    def classify_query_intent(self, query: str, client=None) -> dict:
        # クエリのハッシュ値を生成
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # キャッシュをチェック
        cached_result = self._cached_intent_analysis(query_hash)
        if cached_result:
            return cached_result
        
        # キャッシュになければ通常の処理
        result = super().classify_query_intent(query, client)
        
        # 結果をキャッシュに保存
        self._cached_intent_analysis.cache_clear()  # 古いキャッシュをクリア
        self._cached_intent_analysis(query_hash)
        
        return result
```

## 6. 監視とメトリクス

```python
# 検索品質メトリクスの収集
class SearchMetricsCollector:
    def __init__(self):
        self.metrics = {
            'total_searches': 0,
            'version_filtered': 0,
            'conflict_detected': 0,
            'avg_response_time': 0,
            'cache_hit_rate': 0
        }
    
    def log_search(self, query: str, results: List[dict], 
                   execution_time: float, cache_hit: bool = False):
        """検索実行をログに記録"""
        self.metrics['total_searches'] += 1
        
        # バージョンフィルタリングされた結果の割合
        version_filtered = sum(1 for r in results 
                             if r.get('metadata', {}).get('version_info', {}).get('status') == 'active')
        self.metrics['version_filtered'] += version_filtered / max(len(results), 1)
        
        # 矛盾検出率
        conflicts_found = sum(1 for r in results if r.get('conflicts'))
        if conflicts_found > 0:
            self.metrics['conflict_detected'] += 1
        
        # 平均応答時間の更新
        n = self.metrics['total_searches']
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (n-1) + execution_time) / n
        )
        
        # キャッシュヒット率
        if cache_hit:
            self.metrics['cache_hit_rate'] = (
                (self.metrics['cache_hit_rate'] * (n-1) + 1) / n
            )
        else:
            self.metrics['cache_hit_rate'] = (
                (self.metrics['cache_hit_rate'] * (n-1)) / n
            )
    
    def get_report(self) -> dict:
        """メトリクスレポートを生成"""
        return {
            **self.metrics,
            'conflict_rate': self.metrics['conflict_detected'] / max(self.metrics['total_searches'], 1),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """メトリクスに基づく改善提案を生成"""
        recommendations = []
        
        if self.metrics['avg_response_time'] > 2.0:
            recommendations.append("検索速度が遅いです。インデックスの再構築を検討してください。")
        
        if self.metrics['conflict_rate'] > 0.3:
            recommendations.append("ルール矛盾が多く検出されています。文書の標準化を推進してください。")
        
        if self.metrics['cache_hit_rate'] < 0.2:
            recommendations.append("キャッシュヒット率が低いです。キャッシュサイズの拡大を検討してください。")
        
        return recommendations
```

これらの実装により、以下が実現できます：

1. **最新データの優先表示**: タイムスタンプベースの重み付けとバージョンフィルタリング
2. **ルールの標準化**: 矛盾検出と推奨事項の提示
3. **階層的な検索**: 全社→部門→営業所の優先順位
4. **高速化**: キャッシュとインデックスの最適化
5. **品質監視**: メトリクス収集と改善提案


Codexでコーディングする際は、各メソッドのdocstringとコメントを参考に、必要な部分から段階的に実装してください。

上記実装後に追加機能

-----

## 提案する追記内容

既存の `algorithm.md` は非常に優れているため、その骨格を活かしつつ、以下の4つのセクションを追加・強化します。

### 1\. ユーザーフィードバックによる継続的な精度改善

現在のアルゴリズムは検索結果を提示するところまでですが、ユーザーがその結果に満足したかどうかのフィードバックを収集し、自動で学習・改善していく仕組みを追加します。

**▼ `algorithm.md` に追記するセクション案**

```markdown
## 7. ユーザーフィードバックによる自己改善ループ

検索結果の横に「役に立った」「役に立たなかった」ボタンを設置し、ユーザーからのフィードバックを収集します。

-   **陽性フィードバック（役に立った）**: クリックされた文書とクエリの組み合わせの関連性を強めるように、ベクトルや重みを微調整します。
-   **陰性フィードバック（役に立たなかった）**: なぜ役に立たなかったかの選択肢（例：「情報が古い」「求めている内容と違う」）を提示させ、それを基に問題点を分析します。例えば、「情報が古い」というフィードバックが多ければ、その文書の `recency_weight` の減衰を速めるなどの調整が考えられます。

これにより、システムは使われれば使われるほど賢くなり、メンテナンスの手間を削減できます。
```

-----

### 2\. 「類義語・専門用語」の一元管理

現在、類義語は各チャンクのメタ情報（`meta_info`）に含まれていますが、これをナレッジベース全体で共有・管理する「シソーラス（類義語辞書）」として独立させます。

**▼ `algorithm.md` に追記するセクション案**

```markdown
## 8. 一元管理型シソーラスの導入

会社独自の専門用語や業界用語、略語、新旧の製品名などを一元管理するシソーラス（辞書）機能を導入します。

-   **動的な辞書更新**: ユーザーが検索してもヒットしなかった「ゼロヒットクエリ」を定期的に分析し、新しい類義語の候補として管理者に提示します。
-   **検索への適用**: ユーザーがクエリを入力した際、まずシソーラスを参照して関連用語（例：「稟議」で検索したら「りんぎ」「伺い」「起案」も検索対象に含める）に自動で展開してから検索を実行します。

これにより、ユーザーの言葉の揺らぎを吸収し、検索ヒット率を大幅に向上させることができます。
```

-----

### 3\. ルール矛盾の「予測・未然防止」

現在の矛盾検出は「事後的」ですが、これを一歩進めて「予測的」な機能に強化します。

**▼ `detect_rule_conflicts` メソッドの強化案**

```markdown
### 矛盾の予測と未然防止

文書を新規登録・更新する際に、既存のルールと矛盾が発生しないかを**保存前に**チェックする機能を実装します。

-   **更新時の事前チェック**: ユーザーが新しい就業規則をアップロードしようとした際、システムが「この内容は、既存の全社規定と矛盾する可能性があります。このまま保存しますか？」と警告を表示します。
-   **解決策の提案**: 矛盾を検出した際に、単に警告するだけでなく、「全社規定に合わせて内容を修正する」「部門の例外ルールとして登録する」といった解決策の選択肢を提示します。

これにより、ナレッジベースの健全性をより高いレベルで維持できます。
```

-----

### 4\. 検索アルゴリズムのA/Bテスト

最適な検索パラメータ（重み付けなど）を見つけるために、複数のアルゴリズムを並行して動かし、どちらがより良い結果を出すかを客観的に評価する仕組みを導入します。

**▼ `algorithm.md` に追記するセクション案**

```markdown
## 9. 検索パラメータチューニングのためのA/Bテスト基盤

ユーザーの一部にはアルゴリズムA（例：新しさ重視）を、別の一部にはアルゴリズムB（例：階層重視）を適用し、どちらのアルゴリズムがユーザー満足度（クリック率や「役に立った」率）が高いかを計測します。

-   **自動的な勝者選定**: 一定期間のテストの後、パフォーマンスが良かった方のアルゴリズムが自動で本番環境に適用される仕組みを構築します。

これにより、勘や経験に頼ることなく、データに基づいて検索アルゴリズムを継続的に最適化できます。
```

-----

### まとめ

これらの機能を追加することで、アルゴリズムは、単に高精度な検索システムであるだけでなく、**自ら学習・成長し、データの品質維持まで支援する、より自律的なナレッジマネジメント基盤へと進化するでしょう。**
