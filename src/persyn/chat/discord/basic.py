#!/usr/bin/env python3

import os
import random

import discord
# from discord.ext import commands

TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.message_content = True
# app = commands.app(command_prefix='!', intents=intents)

app = discord.Client(intents=intents)

@app.event
async def on_ready():
    print(f'{app.user} » O N L I N E «')

@app.event
async def on_raw_reaction_add(ctx):
    channel = await app.fetch_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id)

    # msg = await ctx.fetch_message(ctx.message_id)
    print(f'Reaction added: {ctx.message_id} > {ctx.member} : {ctx.emoji} ({message.content})')

@app.event
async def on_raw_reaction_remove(ctx):
    channel = await app.fetch_channel(ctx.channel_id)
    message = await channel.fetch_message(ctx.message_id)
    # msg = await ctx.fetch_message(ctx.message_id)
    print(f'Reaction removed: {ctx.message_id} > {ctx.member} : {ctx.emoji} ({message.content})')


# @app.command()
# async def do_it(message):
#     print(f"> {message.content}")
#     await message.channel.send("Yessir!")

# @app.command()
# async def roll(ctx, dice: str):
#     """Rolls a dice in NdN format."""
#     try:
#         rolls, limit = map(int, dice.split('d'))
#     except Exception:
#         await ctx.send('Format has to be in NdN!')
#         return

#     result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
#     await ctx.send(result)

@app.event
async def on_message(message):
    if message.author == app.user:
        return

    print(f"> {message}")
    await message.channel.send("Ayup.")

@app.event
async def on_error(event, *args, **kwargs):
    print('ERROR:', args)

app.run(TOKEN)
