import csv
from rich.progress import track

from usernames import get_username


class CSVLoader:
    def __init__(self, skip_ids=None):
        self.skip_ids = set(skip_ids) if skip_ids else set()

    def parse(self, posts_filename, comments_filename):
        posts = self._parse_file(posts_filename, True)
        comments = self._parse_file(comments_filename, False)
        threads = self._reconstruct_threads(posts, comments)
        formatted_threads = self._format_threads(threads)
        return formatted_threads

    def _parse_file(self, filename, as_posts=False):
        data = {}
        with open(filename, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in track(
                reader, description=f"Loading {'posts' if as_posts else 'comments'}"
            ):
                if int(row.get("author_id", 0)) not in self.skip_ids:
                    if as_posts:
                        row["comments"] = []
                    else:
                        row["children"] = []
                    data[row["id"]] = row

        return data

    def _reconstruct_threads(self, posts, comments):
        for comment in comments.values():
            if parent_comment := comments.get(comment["parent_comment_id"], None):
                parent_comment["children"].append(comment)
            else:
                if parent_post := posts.get(comment["parent_post"], None):
                    parent_post["comments"].append(comment)

        return {
            post_id: post for post_id, post in posts.items() if int(post_id) != 145556
        }

    def _format_threads(self, threads):
        formatted_data = []
        for post in threads.values():
            thread_tokens = [
                f"{post['sub']}" if "sub" in post else "",
                f"@{get_username(post['id'], post['username'])}",
                post["url"],
                post["title"],
                " ".join(post["body"].split()),
            ]
            for comment in self._iterate_comments(post["comments"]):
                thread_tokens.append(f"@{comment['username']}")
                thread_tokens.append(" ".join(comment["body"].split()))
            formatted_data.append(" ".join(thread_tokens))
        return formatted_data

    def _iterate_comments(self, comments):
        for comment in comments:
            yield comment
            if "children" in comment and comment["children"]:
                yield from self._iterate_comments(comment["children"])

