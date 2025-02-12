---
title: Projects
permalink: /projects/
---

{% for p in collections.posts | reverse %}

<a href="{{p.url}}">{{p.data.title}}</a>

<p style="margin-top" 0;">
<small><em> {{p.data.date | postDate }}</em> - {{p.data.blurb}}</small>
</p>
{% endfor %}
